import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from ogb.nodeproppred import Evaluator
import json
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data import get_dataset, set_train_val_test_split
from graph_rewiring import apply_beltrami
from utils import ROOT_DIR
from oversmooth_metrics import effective_rank, class_mix_score


class BaselineGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(BaselineGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, return_features=False):
        pre_features = x
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        post_features = x
        x = self.convs[-1](x, edge_index)
        
        if return_features:
            return x, pre_features, post_features
        return x


class BaselineGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads=8, dropout=0.5):
        super(BaselineGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(input_dim, output_dim, heads=1, dropout=dropout))
        else:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
    
    def forward(self, x, edge_index, return_features=False):
        pre_features = x
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        post_features = x
        x = self.convs[-1](x, edge_index)
        
        if return_features:
            return x, pre_features, post_features
        return x


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1
    return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def train(model, optimizer, data, opt):
    model.train()
    optimizer.zero_grad()
    feat = data.x
    if opt.get('use_labels', False):
        train_label_idx, train_pred_idx = get_label_masks(data, opt.get('label_rate', 0.5))
        feat = add_labels(feat, data.y, train_label_idx, data.y.max().item() + 1, data.x.device)

    out = model(feat, data.edge_index)

    if opt['dataset'] == 'ogbn-arxiv':
        lf = torch.nn.functional.nll_loss
        loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, opt=None):
    model.eval()
    feat = data.x
    if opt and opt.get('use_labels', False):
        feat = add_labels(feat, data.y, data.train_mask, data.y.max().item() + 1, data.x.device)
    
    logits, accs = model(feat, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def print_model_params(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_params}")

distance_type = 'euclidean'
def run_single_split(opt):
    split_id = opt['split_id']
    dataset_name = opt['dataset']
    time_value = opt['num_layers']
    
    dataset_dir = f'baseline_results/{opt["model_type"]}/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)
    
    opt_split = opt.copy()
    opt_split['split_id'] = split_id
    dataset = get_dataset(opt_split, f'{ROOT_DIR}/data', opt_split.get('not_lcc', True))
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    
    # 获取对应折的mask
    data = dataset.data.clone()
    print(data)
    data.train_mask = data.train_mask[:, split_id].to(torch.bool)
    data.val_mask = data.val_mask[:, split_id].to(torch.bool)
    data.test_mask = data.test_mask[:, split_id].to(torch.bool)
    data = data.to(device)
    print(data)
    
    # 创建模型
    input_dim = data.x.shape[1]
    output_dim = data.y.max().item() + 1
    
    if opt_split['model_type'] == 'GCN':
        model = BaselineGCN(
            input_dim=input_dim,
            hidden_dim=opt_split['hidden_dim'],
            output_dim=output_dim,
            num_layers=opt_split['num_layers'],
            dropout=opt_split['dropout']
        ).to(device)
    elif opt_split['model_type'] == 'GAT':
        model = BaselineGAT(
            input_dim=input_dim,
            hidden_dim=opt_split['hidden_dim'],
            output_dim=output_dim,
            num_layers=opt_split['num_layers'],
            heads=opt_split.get('heads', 8),
            dropout=opt_split['dropout']
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {opt_split['model_type']}")
    
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"\n===== Split {split_id}/9 (single split mode) =====")
    print_model_params(model)
    optimizer = get_optimizer(opt_split['optimizer'], parameters, lr=opt_split['lr'], weight_decay=opt_split['decay'])
    
    best_time = best_epoch = train_acc = val_acc = test_acc = 0
    best_val_metrics = {}
    best_test_metrics = {}
    best_val_model_state = None
    best_val_features = None
    best_test_features = None
    best_val_loss = None
    best_val_acc = 0
    best_test_acc = 0
    losses = []
    this_test = test
    
    for epoch in range(1, opt_split['epoch']):
        start_time = time.time()
        loss = train(model, optimizer, data, opt_split)
        losses.append(loss)
        
        infer_start = time.time()
        tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt_split)
        infer_time = time.time() - infer_start
        
        model.eval()
        with torch.no_grad():
            feat = data.x
            if opt_split.get('use_labels', False):
                feat = add_labels(feat, data.y, data.train_mask, data.y.max().item() + 1, device)
            logits, pre_ode, post_ode = model(feat, data.edge_index, return_features=True)
            
            y_true_val = data.y[data.val_mask].cpu().numpy()
            y_true_test = data.y[data.test_mask].cpu().numpy()
            y_pred_val = logits[data.val_mask].argmax(axis=1).cpu().numpy()
            y_pred_test = logits[data.test_mask].argmax(axis=1).cpu().numpy()
            
            acc_val = accuracy_score(y_true_val, y_pred_val)
            acc_test = accuracy_score(y_true_test, y_pred_test)
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)
            p_test, r_test, f1_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro', zero_division=0)
            
            # 计算特征指标
            pre_ode_val = pre_ode[data.val_mask].float()
            post_ode_val = post_ode[data.val_mask].float()
            effrank_val_before = float(effective_rank(pre_ode_val))
            effrank_val_after = float(effective_rank(post_ode_val))
            effrank_val_ratio = effrank_val_after / (effrank_val_before + 1e-8)
            classmix_val_before = float(class_mix_score(pre_ode_val, data.y[data.val_mask], distance_type=distance_type))
            classmix_val_after = float(class_mix_score(post_ode_val, data.y[data.val_mask], X0=pre_ode_val, distance_type=distance_type))
            classmix_val_ratio = classmix_val_after / (classmix_val_before + 1e-8)
            
            pre_ode_test = pre_ode[data.test_mask].float()
            post_ode_test = post_ode[data.test_mask].float()
            effrank_test_before = float(effective_rank(pre_ode_test))
            effrank_test_after = float(effective_rank(post_ode_test))
            effrank_test_ratio = effrank_test_after / (effrank_test_before + 1e-8)
            classmix_test_before = float(class_mix_score(pre_ode_test, data.y[data.test_mask], distance_type=distance_type))
            classmix_test_after = float(class_mix_score(post_ode_test, data.y[data.test_mask], X0=pre_ode_test, distance_type=distance_type))
            classmix_test_ratio = classmix_test_after / (classmix_test_before + 1e-8)
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_val_metrics = {
                'acc': acc_val,
                'precision': p_val,
                'recall': r_val,
                'f1': f1_val,
                'epoch': epoch,
                'infer_time': infer_time,
                'effrank_before': effrank_val_before,
                'effrank_after': effrank_val_after,
                'effrank_ratio': effrank_val_ratio,
                'classmix_before': classmix_val_before,
                'classmix_after': classmix_val_after,
                'classmix_ratio': classmix_val_ratio
            }
            best_test_metrics = {
                'acc': acc_test,
                'precision': p_test,
                'recall': r_test,
                'f1': f1_test,
                'epoch': epoch,
                'infer_time': infer_time,
                'effrank_before': effrank_test_before,
                'effrank_after': effrank_test_after,
                'effrank_ratio': effrank_test_ratio,
                'classmix_before': classmix_test_before,
                'classmix_after': classmix_test_after,
                'classmix_ratio': classmix_test_ratio
            }
            best_val_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_features = logits[data.val_mask].cpu().numpy()
            best_val_loss = loss
        
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_test_features = logits[data.test_mask].cpu().numpy()
        
        print(f'Epoch: {epoch:03d}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, '
              f'Train: {tmp_train_acc:.4f}, Val: {tmp_val_acc:.4f}, Test: {tmp_test_acc:.4f}')
    
    print(f'best val accuracy {best_val_acc:.3f} with test accuracy {best_test_metrics["acc"]:.3f} '
          f'at epoch {best_val_metrics["epoch"]}')
    
    fold_result = {
        'split': split_id,
        'val': best_val_metrics,
        'test': best_test_metrics,
        'params': {k: v for k, v in opt_split.items()},
        'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_val_loss': best_val_loss
    }
    
    # 保存模型状态
    torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_time{time_value}.pt')
    
    # 保存特征
    if best_val_features is not None:
        np.save(f'{dataset_dir}/val_features_split{split_id}_time{time_value}.npy', best_val_features)
    if best_test_features is not None:
        np.save(f'{dataset_dir}/test_features_split{split_id}_time{time_value}.npy', best_test_features)
    
    with open(f'{dataset_dir}/single_split_summary_time{time_value}.json', 'w') as f:
        json.dump([fold_result], f, indent=2)
    with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'fold_{split_id}'])
        for row in zip(*[losses]):
            writer.writerow(row)
    
    return fold_result, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility.')
    
    # model args
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT'], 
                        help='Model type: GCN or GAT')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for GAT.')
    
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also use labels as features')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
    
    # training args
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs per iteration.')
    parser.add_argument('--time', type=float, default=1.0, help='Time value for file naming.')
    
    # other args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--split_id', type=int, default=None, 
                        help='If set, only run the specified split (0-9) and save summary.json for that split.')

    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt = vars(args)
    
    ten_fold_datasets = [
        'Cora', 'Citeseer', 'Pubmed',
        'Cornell', 'Texas', 'Wisconsin',
        'Chameleon', 'Squirrel'
    ]
    import sys
    result_path = f"./baseline_results/{opt['model_type']}/{opt['dataset']}/fold_results/fold_metrics_time{opt['num_layers']}.json"
    if os.path.exists(result_path):
        print(f"Result already exists at {result_path}, skipping this run.")
        sys.exit(0)
    
    if opt.get('split_id') is not None:
        fold_result, losses = run_single_split(opt)
        exit(0)
    elif opt['dataset'] in ten_fold_datasets:
        num_splits = 10
        results = []
        all_fold_metrics = []
        all_losses = []
        os.makedirs('baseline_results', exist_ok=True)
        
        # 获取数据集
        dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt.get('not_lcc', False))
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        
        # 在mask维度上循环10次
        for split_id in range(num_splits):
            print(f"\n===== Split {split_id+1}/10 =====")
            opt_split = opt.copy()
            opt_split['split_id'] = split_id
            
            fold_result, losses = run_single_split(opt_split)
            all_fold_metrics.append(fold_result)
            all_losses.append(losses)
        
        # 计算并保存汇总结果
        val_accs = [f['val']['acc'] for f in all_fold_metrics]
        test_accs = [f['test']['acc'] for f in all_fold_metrics]
        val_precisions = [f['val']['precision'] for f in all_fold_metrics]
        test_precisions = [f['test']['precision'] for f in all_fold_metrics]
        val_recalls = [f['val']['recall'] for f in all_fold_metrics]
        test_recalls = [f['test']['recall'] for f in all_fold_metrics]
        val_f1s = [f['val']['f1'] for f in all_fold_metrics]
        test_f1s = [f['test']['f1'] for f in all_fold_metrics]
        
        summary = {
            'val_acc_mean': float(np.mean(val_accs)),
            'val_acc_std': float(np.std(val_accs)),
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
            'val_precision_mean': float(np.mean(val_precisions)),
            'val_precision_std': float(np.std(val_precisions)),
            'test_precision_mean': float(np.mean(test_precisions)),
            'test_precision_std': float(np.std(test_precisions)),
            'val_recall_mean': float(np.mean(val_recalls)),
            'val_recall_std': float(np.std(val_recalls)),
            'test_recall_mean': float(np.mean(test_recalls)),
            'test_recall_std': float(np.std(test_recalls)),
            'val_f1_mean': float(np.mean(val_f1s)),
            'val_f1_std': float(np.std(val_f1s)),
            'test_f1_mean': float(np.mean(test_f1s)),
            'test_f1_std': float(np.std(test_f1s)),
            'all_params': opt,
        }
        
        # 保存汇总结果
        dataset_dir = f'baseline_results/{opt["model_type"]}/{opt["dataset"]}/fold_results'
        os.makedirs(dataset_dir, exist_ok=True)
        time_value = opt['num_layers']
        with open(f'{dataset_dir}/fold_metrics_time{time_value}.json', 'w') as f:
            json.dump(all_fold_metrics, f, indent=2)
        with open(f'{dataset_dir}/summary_time{time_value}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'fold_{i}' for i in range(num_splits)])
            for row in zip(*all_losses):
                writer.writerow(row)
        
        print(f"\n===== Final Results =====")
        print(f"Test Accuracy: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
        print(f"Val Accuracy: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
        print(f"Test F1: {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
    else:
        # 单次运行其他数据集
        opt['split_id'] = 0
        fold_result, losses = run_single_split(opt)
        print(f"\n===== Final Results =====")
        print(f"Test Accuracy: {fold_result['test']['acc']:.4f}")
        print(f"Val Accuracy: {fold_result['val']['acc']:.4f}")
        print(f"Test F1: {fold_result['test']['f1']:.4f}") 
from data_handling import get_data
import numpy as np
import torch.optim as optim
from models import *
from torch import nn
from best_params import best_params_dict
import torch
import argparse
import random
import os
import json
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from oversmooth_metrics import effective_rank, class_mix_score

def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_GNN(opt,split):
    data = get_data(opt['dataset'],split)

    best_eval = 10000
    bad_counter = 0
    best_test_acc = 0

    if opt['model'] == 'GraphCON_GCN':
        model = GraphCON_GCN(nfeat=data.num_features,nhid=opt['nhid'],nclass=5,
                             dropout=opt['drop'],nlayers=opt['nlayers'],dt=1.,
                             alpha=opt['alpha'],gamma=opt['gamma'],res_version=opt['res_version']).to(opt['device'])
    elif opt['model'] == 'GraphCON_GAT':
        model = GraphCON_GAT(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                             alpha=opt['alpha'], gamma=opt['gamma'],nheads=opt['nheads']).to(opt['device'])

    optimizer = optim.Adam(model.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay'])
    lf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data), [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(out[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(opt['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(opt['device']))
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model,data)

        if (val_loss < best_eval):
            best_eval = val_loss
            best_test_acc = test_acc
        else:
            bad_counter += 1

        if(bad_counter==opt['patience']):
            break

        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(split, epoch, train_acc, val_acc, test_acc))

    return best_test_acc


def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                json.dumps(v)
            except TypeError as e:
                print(f"Key '{k}' is not serializable: {type(v)} - {e}")
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, range):
        return list(obj)
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}  
    else:
        return obj


def run_single_split(opt):
    """Run training and evaluation for a single split."""
    split_id = opt['split_id']
    dataset_name = opt['dataset']
    time_value = opt['nlayers']
    all_fold_metrics = []
    all_losses = []
    
    dataset_dir = f'results/{dataset_name}_bestopt'
    os.makedirs(dataset_dir, exist_ok=True)
    
    opt_split = opt.copy()
    opt_split['split_id'] = split_id
    data = get_data(opt['dataset'], split_id)
    data = data.clone()
    data.train_mask = data.train_mask[:, split_id].to(torch.bool)
    data.val_mask = data.val_mask[:, split_id].to(torch.bool)
    data.test_mask = data.test_mask[:, split_id].to(torch.bool)
    device = opt['device']
    data = data.to(device)
    
    if opt['model'] == 'GraphCON_GCN':
        model = GraphCON_GCN(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                           dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                           alpha=opt['alpha'], gamma=opt['gamma'], res_version=opt['res_version']).to(device)
    elif opt['model'] == 'GraphCON_GAT':
        model = GraphCON_GAT(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                           dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                           alpha=opt['alpha'], gamma=opt['gamma'], nheads=opt['nheads']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    lf = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_test_acc = 0
    best_val_metrics = {}
    best_test_metrics = {}
    best_val_model_state = None
    best_val_features = None
    best_test_features = None
    best_val_loss = None
    losses = []
    
    for epoch in range(opt['epochs']):
        model.train()
        optimizer.zero_grad()
        out, pre_ode, post_ode = model(data, return_features=True)
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # 评估
        model.eval()
        with torch.no_grad():
            # 记录推理时间
            import time
            start_time = time.time()
            out, pre_ode, post_ode = model(data, return_features=True)
            infer_time = time.time() - start_time
            
            # 计算指标
            y_true_val = data.y[data.val_mask].cpu().numpy()
            y_true_test = data.y[data.test_mask].cpu().numpy()
            y_pred_val = out[data.val_mask].argmax(axis=1).cpu().numpy()
            y_pred_test = out[data.test_mask].argmax(axis=1).cpu().numpy()
            
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
            classmix_val_before = float(class_mix_score(pre_ode_val, data.y[data.val_mask]))
            classmix_val_after = float(class_mix_score(post_ode_val, data.y[data.val_mask], X0=pre_ode_val))
            classmix_val_ratio = classmix_val_after / (classmix_val_before + 1e-8)
            
            pre_ode_test = pre_ode[data.test_mask].float()
            post_ode_test = post_ode[data.test_mask].float()
            effrank_test_before = float(effective_rank(pre_ode_test))
            effrank_test_after = float(effective_rank(post_ode_test))
            effrank_test_ratio = effrank_test_after / (effrank_test_before + 1e-8)
            classmix_test_before = float(class_mix_score(pre_ode_test, data.y[data.test_mask]))
            classmix_test_after = float(class_mix_score(post_ode_test, data.y[data.test_mask], X0=pre_ode_test))
            classmix_test_ratio = classmix_test_after / (classmix_test_before + 1e-8)
        
        # 更新最佳结果
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_val_metrics = {
                'acc': acc_val,
                'precision': p_val,
                'recall': r_val,
                'f1': f1_val,
                'epoch': epoch,
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
            best_val_features = out[data.val_mask].cpu().numpy()
            best_val_loss = loss.item()
        
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_test_features = out[data.test_mask].cpu().numpy()
        
        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val: {acc_val:.4f}, Test: {acc_test:.4f}')
    
    # 保存结果
    fold_result = {
        'split': split_id,
        'val': best_val_metrics,
        'test': best_test_metrics,
        'params': {k: v for k, v in opt_split.items()},
        'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_val_loss': best_val_loss
    }

    fold_results = convert_to_json_serializable(fold_result)
    all_fold_metrics.append(fold_results)
    all_losses.append(losses)
    
    # 保存模型和特征
    # torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_nlayers{time_value}.pt')
    # if best_val_features is not None:
    #     np.save(f'{dataset_dir}/val_features_split{split_id}_nlayers{time_value}.npy', best_val_features)
    # if best_test_features is not None:
    #     np.save(f'{dataset_dir}/test_features_split{split_id}_nlayers{time_value}.npy', best_test_features)
    
    # 保存指标和损失
    with open(f'{dataset_dir}/single_split_summary_nlayers{time_value}.json', 'w') as f:
        json.dump(all_fold_metrics, f, indent=2)
    # with open(f'{dataset_dir}/losses_nlayers{time_value}.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['loss'])
    #     for loss in losses:
    #         writer.writerow([loss])
    
    return best_test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='texas',
                        help='cornell, wisconsin, texas')
    parser.add_argument('--model', type=str, default='GraphCON_GCN',
                        help='GraphCON_GCN, GraphCON_GAT')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter of graphCON')
    parser.add_argument('--gamma', type=float, default=1.,
                        help='gamma parameter of graphCON')
    parser.add_argument('--nheads', type=int, default=4,
                        help='number of attention heads for GraphCON-GAT')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--res_version', type=int, default=1,
                        help='version of residual connection')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--device', type=str, default=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='number of splits')
    parser.add_argument('--split_id', type=int, default=None, help='If set, only run the specified split')
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed')

    args = parser.parse_args()
    opt = vars(args)

    try:
        best_opt = best_params_dict[opt['dataset']]
        opt = {**best_opt, **opt}
        print(f"Using best opt for {opt['dataset']}")
    except KeyError:
        opt = opt
    seed_all(opt['seed'])

    print(opt)

    if opt.get('split_id') is not None:
        seed_all(opt['seed'])
        best = run_single_split(opt)
        print('Test accuracy: ', best*100)
    else:
        seed_all(opt['seed'])
        n_splits = opt['n_splits']
        all_fold_metrics = []
        all_losses = []
        dataset_dir = f'results/{opt["dataset"]}_bestopt'
        os.makedirs(dataset_dir, exist_ok=True)
        time_value = opt['nlayers']
        
        for split_id in range(n_splits):
            print(f"\n===== Split {split_id+1}/{n_splits} =====")
            opt_split = opt.copy()
            opt_split['split_id'] = split_id
            best = run_single_split(opt_split)
            
            # 加载这一折的结果
            with open(f'{dataset_dir}/single_split_summary_nlayers{time_value}.json', 'r') as f:
                fold_metrics = json.load(f)
                all_fold_metrics.extend(fold_metrics)
            
            # with open(f'{dataset_dir}/losses_nlayers{time_value}.csv', 'r') as f:
            #     reader = csv.reader(f)
            #     next(reader)  # Skip header
            #     losses = [float(row[0]) for row in reader]
            #     all_losses.append(losses)
        
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
        summary = convert_to_json_serializable(summary)
        with open(f'{dataset_dir}/summary_nlayers{time_value}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(f'{dataset_dir}/fold_metrics_nlayers{time_value}.json', 'w') as f:
            json.dump(all_fold_metrics, f, indent=2)
        with open(f'{dataset_dir}/losses_nlayers{time_value}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'fold_{i}' for i in range(n_splits)])
            for row in zip(*all_losses):
                writer.writerow(row)
        
        print('Mean test accuracy: ', np.mean(np.array(test_accs)*100), 'std: ', np.std(np.array(test_accs)*100))

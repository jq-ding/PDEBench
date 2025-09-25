from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
# import dgl.graph_index
import numpy as np
import torch
from config import parser
# from utils.data_utils import get_dataset
from utils.train_utils import get_dir_name, format_metrics
import scipy.sparse as sp
# import dgl
import statistics

import random
from layers.ham_layers_v1 import HamGraphConvolution
# from prettytable import PrettyTable
import sys
from torch_geometric.utils import get_laplacian,to_dense_adj,to_scipy_sparse_matrix,add_remaining_self_loops
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from utils.data_utils_lp import load_data
from utils.eval_utils import acc_f1
from utils.oversmooth_metrics import effective_rank, class_mix_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv

class HamGNN(torch.nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_classes, num_layers):
        super(HamGNN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # self.out_dim = out_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = args.dropout

        self.linear_layer1 = torch.nn.Linear(in_dim, hidden_dim)
        self.liner_layer2 = torch.nn.Linear(hidden_dim, num_classes, bias=False)
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()
        # self.layers.append(HamGraphConvolution(hidden_dim,args))
        
        for i in range(num_layers):
            self.layers.append(HamGraphConvolution(hidden_dim,args))
            # self.layers_bn.append(torch.nn.BatchNorm1d(hidden_dim))
        # self.layers.append(HamGraphConvolution(hidden_dim, out_dim, num_classes, dropout, use_cuda))

        # self.bn_input = torch.nn.BatchNorm1d(hidden_dim)

        if not args.act:
            self.act = lambda x: x
        elif args.act == 'elu':
            self.act = F.elu
        else:
            self.act = getattr(torch, args.act)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

    def forward(self, features, adj, return_features=False):
        # h = F.dropout(features, p=self.dropout, training=self.training)
        h = features
        h = self.linear_layer1(h)
        # h = self.bn_input(h)
        pre_ode_features = h.detach()
        for i, layer in enumerate(self.layers):
            h = layer(h,adj)

            # h = self.layers_bn[i](h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.act(h)
        post_ode_features = h.detach()

        h = self.liner_layer2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if return_features:
            return h, pre_ode_features, post_ode_features
        return h

    def compute_metrics(self, embeddings, data, split):
        idx = data['splits'][split]
        output = F.log_softmax(embeddings[idx], dim=1)
        loss = F.nll_loss(output, data['labels'][idx])
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_model_params(model):
  print(model)
#   table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  ham_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
    #   table.add_row([name, params])
      total_params += params
#   print(table)
  print(f"Total Trainable Params: {total_params}")

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

def test( model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index,data.edge_weight)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs

def run_single_split(opt):
    split_id = opt['split_id']
    dataset_name = opt['dataset']
    time_value = opt['num_layers']
    all_fold_metrics = []
    all_losses = []
    all_best_val_features = []
    
    dataset_dir = f'results/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)
    
    opt_split = opt.copy()
    opt_split['split_id'] = split_id
    opt_split['task'] = 'nc' 
    data = load_data(opt_split, os.path.join('./data', opt_split['dataset']))
    
    # 获取对应折的mask
    idx_train = data['splits'][split_id]['train']
    idx_val = data['splits'][split_id]['val']
    idx_test = data['splits'][split_id]['test']
    
    args.n_nodes, args.feat_dim = data['features'].shape
    args.n_classes = data['labels'].max().item() + 1
    model = HamGNN(args, args.feat_dim, args.hidden, args.n_classes , args.num_layers).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    print(str(model))

    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    criterion = torch.nn.CrossEntropyLoss()
    best_time = best_epoch = train_acc = val_acc = test_acc = counter = 0
    best_val_metrics = {}
    best_test_metrics = {}
    best_val_model_state = None
    best_val_features = None
    best_test_features = None
    best_val_loss = None
    best_val_acc = 0
    best_test_acc = 0
    losses = []

    data['features'] = data['features'].to(args.device)
    data['adj_train_norm'] = data['adj_train_norm'].to(args.device)
    data['labels'] = data['labels'].to(args.device)
    
    for epoch in range(1, opt['epoch']):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(data['features'], data['adj_train_norm'])
        
        # 使用对应折的索引计算损失
        train_loss = criterion(out[idx_train], data['labels'][idx_train])
        train_loss.backward()
        optimizer.step()
        
        # 计算训练集准确率
        train_pred = out[idx_train].argmax(dim=1)
        train_acc = (train_pred == data['labels'][idx_train]).float().mean()
        
        # 计算验证集和测试集准确率
        model.eval()
        with torch.no_grad():
            val_pred = out[idx_val].argmax(dim=1)
            val_acc = (val_pred == data['labels'][idx_val]).float().mean()
            
            test_pred = out[idx_test].argmax(dim=1)
            test_acc = (test_pred == data['labels'][idx_test]).float().mean()
            
            # 计算特征指标
            logits, pre_ode, post_ode = model(data['features'], data['adj_train_norm'], return_features=True)
            
            # 计算指标
            y_true_val = data['labels'][idx_val].cpu().numpy()
            y_true_test = data['labels'][idx_test].cpu().numpy()
            y_pred_val = logits[idx_val].argmax(axis=1).cpu().numpy()
            y_pred_test = logits[idx_test].argmax(axis=1).cpu().numpy()
            
            acc_val = accuracy_score(y_true_val, y_pred_val)
            acc_test = accuracy_score(y_true_test, y_pred_test)
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)
            p_test, r_test, f1_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro', zero_division=0)
            
            # 计算特征指标
            pre_ode_val = pre_ode[idx_val].float()
            post_ode_val = post_ode[idx_val].float()
            effrank_val_before = float(effective_rank(pre_ode_val))
            effrank_val_after = float(effective_rank(post_ode_val))
            effrank_val_ratio = effrank_val_after / (effrank_val_before + 1e-8)
            classmix_val_before = float(class_mix_score(pre_ode_val, data['labels'][idx_val], distance_type='euclidean'))
            classmix_val_after = float(class_mix_score(post_ode_val, data['labels'][idx_val], X0=pre_ode_val, distance_type='euclidean'))
            classmix_val_ratio = classmix_val_after / (classmix_val_before + 1e-8)
            
            pre_ode_test = pre_ode[idx_test].float()
            post_ode_test = post_ode[idx_test].float()
            effrank_test_before = float(effective_rank(pre_ode_test))
            effrank_test_after = float(effective_rank(post_ode_test))
            effrank_test_ratio = effrank_test_after / (effrank_test_before + 1e-8)
            classmix_test_before = float(class_mix_score(pre_ode_test, data['labels'][idx_test], distance_type='euclidean'))
            classmix_test_after = float(class_mix_score(post_ode_test, data['labels'][idx_test], X0=pre_ode_test, distance_type='euclidean'))
            classmix_test_ratio = classmix_test_after / (classmix_test_before + 1e-8)
        
        losses.append(train_loss.item())
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_val_metrics = {
                'acc': acc_val,
                'precision': p_val,
                'recall': r_val,
                'f1': f1_val,
                'epoch': epoch,
                'infer_time': time.time() - start_time,
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
                'infer_time': time.time() - start_time,
                'effrank_before': effrank_test_before,
                'effrank_after': effrank_test_after,
                'effrank_ratio': effrank_test_ratio,
                'classmix_before': classmix_test_before,
                'classmix_after': classmix_test_after,
                'classmix_ratio': classmix_test_ratio
            }
            best_val_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_features = logits[idx_val].cpu().numpy()
            best_val_loss = train_loss.item()
        
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_test_features = logits[idx_test].cpu().numpy()
        
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(train_loss.item()),
              'acc_train: {:.4f}'.format(train_acc.item()),
              'acc_val: {:.4f}'.format(val_acc.item()),
              'acc_test: {:.4f}'.format(test_acc.item()),
              'time: {:.4f}s'.format(time.time() - start_time))
        
        if val_acc > best_val_acc:
            best_time = time.time() - start_time
            best_epoch = epoch
            train_acc = train_acc.item()
            val_acc = val_acc.item()
            test_acc = test_acc.item()
            counter = 0
        else:
            counter += 1
        # if counter == args.patience:
        #     print('Early stopping!')
        #     break
    
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch),
            "train_acc= {:.4f}".format(train_acc),
            "val_acc= {:.4f}".format(val_acc),
            "test_acc= {:.4f}".format(test_acc),
            "time= {:.4f}s".format(best_time))
    
    fold_result = {
        'split': split_id,
        'val': best_val_metrics,
        'test': best_test_metrics,
        'params': {k: v for k, v in opt_split.items()},
        'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_val_loss': best_val_loss
    }
    
    # 保存模型状态
    torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_nlayers{time_value}.pt')
    
    # 保存特征
    if best_val_features is not None:
        np.save(f'{dataset_dir}/val_features_split{split_id}_nlayers{time_value}.npy', best_val_features)
    if best_test_features is not None:
        np.save(f'{dataset_dir}/test_features_split{split_id}_nlayers{time_value}.npy', best_test_features)
    
    # 保存每一折的结果
    with open(f'{dataset_dir}/single_split_summary_nlayers{time_value}.json', 'w') as f:
        json.dump([fold_result], f, indent=2)
    with open(f'{dataset_dir}/losses_nlayers{time_value}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'fold_{split_id}'])
        for row in zip(*[losses]):
            writer.writerow(row)
    
    return test_acc, fold_result, losses

if __name__ == '__main__':
    args = parser.parse_args()
    test_acc_list = []
    
    # create log file
    if not os.path.exists('./log'):
        os.makedirs('./log')
    # add time stamp to log file name to avoid overwriting
    time_stamp = time.strftime("%H%M%S", time.localtime())
    log_file = './log/' + args.dataset + '_' + str(args.odemap) + '_' + str(args.num_layers) + '_' + str(args.agg) + '_' + time_stamp + '.txt'

    # write command line args to log file
    with open(log_file, 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    result_path = f"./results/{args.dataset}/fold_metrics_nlayers{args.num_layers}.json"
    if os.path.exists(result_path):
        print(f"Result already exists at {result_path}, skipping this run.")
        sys.exit(0)

    if args.num_splits == 1:
        # 单折运行
        args.split_id = 0  # 使用第一折的数据
        args.task = 'nc'  # 添加task参数
        test_acc, fold_result, losses = run_single_split(vars(args))
        test_acc_list.append(test_acc.cpu().numpy())
        print("=====================================")
        print("test_acc: ", test_acc)
        # write log
        with open(log_file, 'a') as f:
            f.write('test_acc: ' + str(test_acc) + ' ')
            f.write('\n')
        print("test_acc_list: ", test_acc_list)
        print("mean: ", np.mean(test_acc_list))
        print("std: ", np.std(test_acc_list))
    else:
        # 十折交叉验证
        all_fold_metrics = []
        all_losses = []
        dataset_dir = f'results/{args.dataset}'
        os.makedirs(dataset_dir, exist_ok=True)
        
        for split_id in range(10):  # 10折交叉验证
            args.split_id = split_id
            args.task = 'nc'  # 添加task参数
            test_acc, fold_result, losses = run_single_split(vars(args))
            test_acc_list.append(float(test_acc))
            all_fold_metrics.append(fold_result)
            all_losses.append(losses)
            print("=====================================")
            print("split: ", split_id)
            print("test_acc: ", test_acc)
            # write log
            with open(log_file, 'a') as f:
                f.write('split: ' + str(split_id) + ' test_acc: ' + str(test_acc) + ' ')
                f.write('\n')
            print("test_acc_list: ", test_acc_list)
            print("mean: ", np.mean(test_acc_list))
            print("std: ", np.std(test_acc_list))
        
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
            'all_params': vars(args),
        }
        
        # 保存汇总结果
        time_value = args.num_layers
        with open(f'{dataset_dir}/fold_metrics_nlayers{time_value}.json', 'w') as f:
            json.dump(all_fold_metrics, f, indent=2)
        with open(f'{dataset_dir}/summary_nlayers{time_value}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(f'{dataset_dir}/losses_nlayers{time_value}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'fold_{i}' for i in range(10)])
            for row in zip(*all_losses):
                writer.writerow(row)
    
    # write log
    with open(log_file, 'a') as f:
        f.write('test_acc_list: ' + str(test_acc_list) + ' ')
        f.write('\n')
        f.write('mean: ' + str(np.mean(test_acc_list)) + ' ')
        f.write('\n')
        f.write('std: ' + str(np.std(test_acc_list)) + ' ')
        f.write('\n')
        #dump args dict  to log
        json.dump(vars(args), f, indent=4)
    # change saved log file name to include mean
    os.rename(log_file, log_file[:-4] + '_mean_' + str(np.mean(test_acc_list)) + '.txt')






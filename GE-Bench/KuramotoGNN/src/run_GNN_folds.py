import argparse
import numpy as np
import torch
import gc
import os
import wandb
import json
import csv
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
import time
from data import get_dataset, set_train_val_test_split
from data_heterophilic import get_data
from ogb.nodeproppred import Evaluator
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
from best_params import  best_params_dict
from oversmooth_metrics import effective_rank, class_mix_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

# # 设置环境变量以确保确定性
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True)

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


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out, omega = model(feat, pos_encoding)
  # omega = omega.mean(dim=-1)
  # cos_state = torch.mean(torch.cos(omega))
  # sin_state = torch.mean(torch.sin(omega))
  # order = torch.sqrt(cos_state**2 + sin_state**2)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  torch.cuda.empty_cache()
  return loss.item()


def train_OGB(model, mp, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  pos_encoding = mp(pos_encoding).to(model.device)
  out, omega = model(feat, pos_encoding)
  omega_norm = 0
  for i in range(omega.size(0)):
    for j in range(i+1, omega.size(0)):
      omega_norm += torch.norm(omega[i]-omega[j])
  
  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, omega = model(feat, pos_encoding)
  omega = omega.mean(dim=-1)
  cos_state = torch.mean(torch.cos(omega))
  sin_state = torch.mean(torch.sin(omega))
  order = torch.sqrt(cos_state**2 + sin_state**2)
  
  accs = []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  # accs.append(order)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, pos_encoding, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()

  out = model(feat, pos_encoding)[0].log_softmax(dim=-1)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[data.train_mask],
    'y_pred': y_pred[data.train_mask],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[data.val_mask],
    'y_pred': y_pred[data.val_mask],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[data.test_mask],
    'y_pred': y_pred[data.test_mask],
  })['acc']

  return train_acc, valid_acc, test_acc


def run_single_split(opt):
    """Run training and evaluation for a single split."""
    set_seed(opt['seed'])

    try:
        best_opt = best_params_dict[opt['dataset']]
        opt = {**best_opt, **opt}
        print("Using best params for ", opt['dataset'])
    except KeyError:
        opt = opt

    if opt['kuramoto']==1:
      opt['time'] = opt['time']
      opt['step_size'] = opt['step_size']
      opt['method'] = opt['method']
      opt['hidden_dim'] = opt['hidden_dim']
      opt['add_source'] = False
    else:
      opt['time'] = opt['time']
      opt['step_size'] = opt['step_size']
      opt['method'] = opt['method']
    
    dataset = get_dataset(opt, '../data', opt.get('not_lcc', False))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get corresponding fold's mask
    data = dataset.data.clone()
    split_id = opt['split_id']
    data.train_mask = data.train_mask[:, split_id].to(torch.bool)
    data.val_mask = data.val_mask[:, split_id].to(torch.bool)
    data.test_mask = data.test_mask[:, split_id].to(torch.bool)
    data = data.to(device)
    
    if opt['beltrami']:
        pos_encoding = apply_beltrami(data, opt).to(device)
        opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
        pos_encoding = None
        
    if opt['rewire_KNN'] or opt['fa_layer']:
        model = GNN_KNN(opt, dataset, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, dataset, device).to(device)
    else:
        model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)
    
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
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
    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
    
    # Training loop
    for epoch in range(1, opt['epoch']):
        start_time = time.time()
        if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
            ei = apply_KNN(data, pos_encoding, model, opt)
            model.odeblock.odefunc.edge_index = ei
            
        loss = train(model, optimizer, data, pos_encoding)
        losses.append(loss)
        
        infer_start = time.time()
        tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)
        infer_time = time.time() - infer_start
        
        # Evaluation and feature extraction
        model.eval()
        with torch.no_grad():
            feat = data.x
            if model.opt['use_labels']:
                feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
            logits, pre_ode, post_ode = model(feat, pos_encoding, return_features=True)
            
            # Calculate metrics
            y_true_val = data.y[data.val_mask].cpu().numpy()
            y_true_test = data.y[data.test_mask].cpu().numpy()
            y_pred_val = logits[data.val_mask].argmax(axis=1).cpu().numpy()
            y_pred_test = logits[data.test_mask].argmax(axis=1).cpu().numpy()
            
            acc_val = accuracy_score(y_true_val, y_pred_val)
            acc_test = accuracy_score(y_true_test, y_pred_test)
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)
            p_test, r_test, f1_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro', zero_division=0)
            
            # Calculate oversmooth metrics
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
        
        # Update best results
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
        
        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'
        print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
    
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(
        best_val_acc, best_test_metrics['acc'], best_val_metrics['epoch'], best_time))
    
    # Save results
    dataset_dir = f'results/{opt["dataset"]}_bestopt'
    os.makedirs(dataset_dir, exist_ok=True)
    time_value = opt['time']
    
    # Save model and features
    torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_time{time_value}.pt')
    if best_val_features is not None:
        np.save(f'{dataset_dir}/val_features_split{split_id}_time{time_value}.npy', best_val_features)
    if best_test_features is not None:
        np.save(f'{dataset_dir}/test_features_split{split_id}_time{time_value}.npy', best_test_features)
    
    # Save metrics
    fold_result = {
        'split': split_id,
        'val': best_val_metrics,
        'test': best_test_metrics,
        'params': {k: v for k, v in opt.items()},
        'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_val_loss': best_val_loss
    }
    
    with open(f'{dataset_dir}/single_split_summary_time{time_value}.json', 'w') as f:
        json.dump([fold_result], f, indent=2)
    
    # Save losses
    with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss'])
        for loss in losses:
            writer.writerow([loss])

def main(cmd_opt):
  try:
      best_opt = best_params_dict[cmd_opt['dataset']]
      opt = {**best_opt, **cmd_opt}
      print("Using best params in main for ", opt['dataset'])
  except KeyError:
      opt = cmd_opt

  if opt['kuramoto']==1:
    opt['time'] = cmd_opt['time']
    opt['step_size'] = cmd_opt['step_size']
    opt['method'] = cmd_opt['method']
    opt['hidden_dim'] = cmd_opt['hidden_dim']
    opt['add_source'] = False
  else:
    opt['time'] = cmd_opt['time']
    opt['step_size'] = cmd_opt['step_size']
    opt['method'] = cmd_opt['method']
    # opt = cmd_opt

  set_seed(opt['seed'])

  # List of datasets that support ten-fold cross-validation
  ten_fold_datasets = [
      'Cora', 'Citeseer', 'Pubmed',
      'Cornell', 'Texas', 'Wisconsin',
      'Chameleon', 'Squirrel'
  ]
  print("Running for ", opt['dataset'])
  if opt.get('split_id') is not None:
      # Run single split
      print("Running single split for ", opt['dataset'])
      run_single_split(opt)
  elif opt['dataset'] in ten_fold_datasets:
      # Run ten-fold cross-validation
      print("Running ten-fold cross-validation for ", opt['dataset'])
      num_splits = 10
      all_fold_metrics = []
      all_losses = []
      dataset_dir = f'results/{opt["dataset"]}_bestopt'
      os.makedirs(dataset_dir, exist_ok=True)
      time_value = opt['time']

      for split_id in range(num_splits):
          print(f"\n===== Split {split_id+1}/10 =====")
          opt_split = opt.copy()
          opt_split['split_id'] = split_id
          run_single_split(opt_split)

          # Load the saved metrics for this split
          with open(f'{dataset_dir}/single_split_summary_time{time_value}.json', 'r') as f:
              fold_metrics = json.load(f)
              all_fold_metrics.extend(fold_metrics)

          # Load the saved losses for this split
          with open(f'{dataset_dir}/losses_time{time_value}.csv', 'r') as f:
              reader = csv.reader(f)
              next(reader)  # Skip header
              losses = [float(row[0]) for row in reader]
              all_losses.append(losses)

      # Calculate and save summary metrics
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

      # Save summary results
      with open(f'{dataset_dir}/summary_time{time_value}.json', 'w') as f:
          json.dump(summary, f, indent=2)

      # Save all fold metrics
      with open(f'{dataset_dir}/fold_metrics_time{time_value}.json', 'w') as f:
          json.dump(all_fold_metrics, f, indent=2)

      # Save all losses
      with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
          writer = csv.writer(f)
          writer.writerow([f'fold_{i}' for i in range(num_splits)])
          for row in zip(*all_losses):
              writer.writerow(row)
  # else:
  #     # Run original main function for other datasets
  #     main(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # KuGraph args
    parser.add_argument('--coupling_strength', type=float, default=3.0, help='Kuramoto coupling strength')
    parser.add_argument('--run_time', type=int, default=1, help='The current number of runs')  

    parser.add_argument('--split_rate', type=int, default=20, help='The current number of runs')  
    parser.add_argument('--kuramoto', type=int, default=1, help='Run the kuramotoGNN or not') 
    parser.add_argument('--add_noise', type=int, default=0, help='Add noise option')
    parser.add_argument('--logging', type=int, default=0, help='Do the wandb logging or not')
    parser.add_argument('--split_hetero', type=int, default=0, help='The current number of runs')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                            "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64", help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance", help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention", help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    parser.add_argument('--symmetric_attention', action='store_true', help='maks the attention symmetric for rewring in QK space')


    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--discritize_type', type=str, default="norm")
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility.')
    parser.add_argument('--split_id', type=int, default=None, help='If set, only run the specified split (0-9) and save summary.json for that split.')
    args = parser.parse_args()

    # 在脚本开始时设置随机种子
    set_seed(args.seed)
    
    opt = vars(args)

    import sys  
    result_path = f"./results/{opt['dataset']}_bestopt/fold_metrics_time{opt['time']}.json"
    # if os.path.exists(result_path) or os.path.exists(f"./results/{opt['dataset']}_bestopt/all_fold_metrics_time{opt['time']}.json"):
    #     print(f"Result already exists at {result_path}, skipping this run.")
    #     sys.exit(0)

    opt['is_webKB'] = False
    if opt.get('split_id') is not None:
        run_single_split(opt)
    else:
        main(opt)

import argparse
import time
import os
import random
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
import json
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from GNN import GNN
from GNN_early import GNNEarly
# from GNN_KNN import GNN_KNN
# from GNN_KNN_early import GNNKNNEarly
from data import get_dataset, set_train_val_test_split
# from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
# from best_params import best_params_dict
# from heterophilic import get_fixed_splits
# from utils import ROOT_DIR
# from CGNN import CGNN, get_sym_adj
# from CGNN import train as train_cgnn
from oversmooth_metrics import effective_rank, class_mix_score

from good_params_graphCON import good_params_dict
import wandb


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


def train(model, optimizer, data):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out = model(feat)
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
  out = model(feat, pos_encoding)

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
def test(model, data, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
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

  out = model(feat, pos_encoding).log_softmax(dim=-1)
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


def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['function'] is not None:
        opt['function'] = cmd_opt['function']
    if cmd_opt['block'] is not None:
        opt['block'] = cmd_opt['block']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['method'] is not None:
        opt['method'] = cmd_opt['method']
    if cmd_opt['step_size'] != 1:
        opt['step_size'] = cmd_opt['step_size']
    if cmd_opt['time'] != 1:
        opt['time'] = cmd_opt['time']
    if cmd_opt['epoch'] != 100:
        opt['epoch'] = cmd_opt['epoch']

    if not cmd_opt['not_lcc']:
        opt['not_lcc'] = False
    if cmd_opt['num_splits'] != 1:
        opt['num_splits'] = cmd_opt['num_splits']


def main(cmd_opt):
    try:
        best_opt = good_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    except KeyError:
        opt = cmd_opt

    dataset = get_dataset(opt, '../../data', opt['not_lcc'])
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)

    data = dataset.data.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0

    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test

    for epoch in range(1, opt['epoch']):
        start_time = time.time()

        loss = train(model, optimizer, data, pos_encoding)
        tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

        if tmp_val_acc > val_acc:
            best_epoch = epoch
            train_acc = tmp_train_acc
            val_acc = tmp_val_acc
            test_acc = tmp_test_acc
            if not opt['no_early']:
                best_time = model.odeblock.test_integrator.solver.best_time
            else:
                best_time = opt['time']
            if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
                best_epoch = epoch
                val_acc = model.odeblock.test_integrator.solver.best_val
                test_acc = model.odeblock.test_integrator.solver.best_test
                train_acc = model.odeblock.test_integrator.solver.best_train
                best_time = model.odeblock.test_integrator.solver.best_time

                log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

        print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc, best_time))
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
    return train_acc, val_acc, test_acc

distance_type = 'euclidean'
def run_single_split(opt):
    # try:
    #     best_opt = good_params_dict[opt['dataset']]
    #     opt = {**opt, **best_opt}
    #     merge_cmd_args(opt, opt)
    # except KeyError:
    #     opt = opt
    split_id = opt['split_id']
    dataset_name = opt['dataset']
    time_value = opt['time']
    all_fold_metrics = []
    all_losses = []
    all_best_val_features = []
    
    dataset_dir = f'results/{dataset_name}_bestopt'
    os.makedirs(dataset_dir, exist_ok=True)
    
    opt_split = opt.copy()
    opt_split['split_id'] = split_id
    dataset = get_dataset(opt_split, '../../data', opt_split.get('not_lcc', True))
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    data = dataset.data
    data.train_mask = data.train_mask[:, split_id].to(torch.bool)
    data.val_mask = data.val_mask[:, split_id].to(torch.bool)
    data.test_mask = data.test_mask[:, split_id].to(torch.bool)
    data = data.to(device)
    print(data)
    
    model = GNN(opt_split, dataset, device).to(device) if opt_split["no_early"] else GNNEarly(opt_split, dataset, device).to(device)
    
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
    this_test = test_OGB if opt_split['dataset'] == 'ogbn-arxiv' else test
    for epoch in range(1, opt_split['epoch']):
      start_time = time.time()
      loss = train(model, optimizer, data)
      losses.append(loss)
      infer_start = time.time()
      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt_split)
      infer_time = time.time() - infer_start
      model.eval()
      with torch.no_grad():
        feat = data.x
        if model.opt['use_labels']:
          feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
        logits, pre_ode, post_ode = model(feat, return_features=True)
        y_true_val = data.y[data.val_mask].cpu().numpy()
        y_true_test = data.y[data.test_mask].cpu().numpy()
        y_pred_val = logits[data.val_mask].argmax(axis=1).cpu().numpy()
        y_pred_test = logits[data.test_mask].argmax(axis=1).cpu().numpy()
        acc_val = accuracy_score(y_true_val, y_pred_val)
        acc_test = accuracy_score(y_true_test, y_pred_test)
        # val
        p_val, r_val, f1_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro', zero_division=0)
        p_test, r_test, f1_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro', zero_division=0)
        pre_ode_val = pre_ode[data.val_mask].float()
        post_ode_val = post_ode[data.val_mask].float()
        effrank_val_before = float(effective_rank(pre_ode_val))
        effrank_val_after = float(effective_rank(post_ode_val))
        effrank_val_ratio = effrank_val_after / (effrank_val_before + 1e-8)
        classmix_val_before = float(class_mix_score(pre_ode_val, data.y[data.val_mask], distance_type=distance_type))
        classmix_val_after = float(class_mix_score(post_ode_val, data.y[data.val_mask], X0=pre_ode_val, distance_type=distance_type))
        classmix_val_ratio = classmix_val_after / (classmix_val_before + 1e-8)
        # test
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
      log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'
      print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(best_val_acc, best_test_metrics['acc'], best_val_metrics['epoch'], best_time))
    fold_result = {
      'split': split_id,
      'val': best_val_metrics,
      'test': best_test_metrics,
      'params': {k: v for k, v in opt_split.items()},
      'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
      'best_val_loss': best_val_loss
    }
    all_fold_metrics.append(fold_result)
    all_losses.append(losses)
    
    # 保存模型状态
    torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_time{time_value}.pt')
    
    # 保存特征
    if best_val_features is not None:
      np.save(f'{dataset_dir}/val_features_split{split_id}_time{time_value}.npy', best_val_features)
    if best_test_features is not None:
      np.save(f'{dataset_dir}/test_features_split{split_id}_time{time_value}.npy', best_test_features)
    
    # 打印调试信息
    print("\nDebug: Content of all_fold_metrics before saving:")
    print(json.dumps(all_fold_metrics, indent=2))
    
    with open(f'{dataset_dir}/single_split_summary_time{time_value}.json', 'w') as f:
      json.dump(all_fold_metrics, f, indent=2)
    with open(f'{dataset_dir}/losses_time{time_value}.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([f'fold_{split_id}'])
      for row in zip(*all_losses):
        writer.writerow(row)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility.')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                      help='use the 10 fixed splits from '
                           'https://arxiv.org/abs/2002.05287')
  parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                      help='the number of splits to repeat the results on')
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
  parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs per iteration.')
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
  parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

  # ODE args
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=200, help='maximum number of integration steps')
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
  parser.add_argument("--max_nfe", type=int, default=10000,
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
  parser.add_argument('--pos_enc_type', type=str, default="DW64",
                      help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                      help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention",
                      help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  parser.add_argument('--symmetric_attention', action='store_true',
                      help='maks the attention symmetric for rewring in QK space')

  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

  parser.add_argument('--split_id', type=int, default=None, help='If set, only run the specified split (0-9) and save summary.json for that split.')

  parser.add_argument('--id', type=int)
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
  try:
    best_opt = good_params_dict[opt['dataset']]
    opt = {**best_opt, **opt}
  except KeyError:
    opt = opt

  ten_fold_datasets = [
    'Cora', 'Citeseer', 'Pubmed',
    'Cornell', 'Texas', 'Wisconsin',
    'Chameleon', 'Squirrel'
  ]
  if opt.get('split_id') is not None:
    run_single_split(opt)
    exit(0)
  elif opt['dataset'] in ten_fold_datasets:
    num_splits = 10
    results = []
    all_fold_metrics = []
    all_losses = []
    all_best_val_features = []
    os.makedirs('results', exist_ok=True)
    
    # 获取数据集
    dataset = get_dataset(opt, '../../data', opt.get('not_lcc', False))
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    print("Running 10 folds")
    # 在mask维度上循环10次
    for split_id in range(num_splits):
        print(f"\n===== Split {split_id+1}/10 =====")
        opt_split = opt.copy()
        opt_split['split_id'] = split_id
        
        # 获取对应折的mask
        data = dataset.data.clone()
        data.train_mask = data.train_mask[:, split_id].to(torch.bool)
        data.val_mask = data.val_mask[:, split_id].to(torch.bool)
        data.test_mask = data.test_mask[:, split_id].to(torch.bool)
        data = data.to(device)
        
        if opt_split['beltrami']:
            pos_encoding = apply_beltrami(data, opt_split).to(device)
            opt_split['pos_enc_dim'] = pos_encoding.shape[1]
        else:
            pos_encoding = None
            
        model = GNN(opt_split, dataset, device).to(device) if opt_split["no_early"] else GNNEarly(opt_split, dataset, device).to(device)
        
        parameters = [p for p in model.parameters() if p.requires_grad]
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
        this_test = test_OGB if opt_split['dataset'] == 'ogbn-arxiv' else test
        
        # 训练循环
        for epoch in range(1, opt_split['epoch']):
            start_time = time.time()
            loss = train(model, optimizer, data)
            losses.append(loss)
            
            infer_start = time.time()
            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt_split)
            infer_time = time.time() - infer_start
            
            # 评估和特征提取
            model.eval()
            with torch.no_grad():
                feat = data.x
                if model.opt['use_labels']:
                    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
                logits, pre_ode, post_ode = model(feat, return_features=True)
                
                # 计算指标
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
            
            # 更新最佳结果
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
        
        # 保存每一折的结果
        fold_result = {
            'split': split_id,
            'val': best_val_metrics,
            'test': best_test_metrics,
            'params': {k: v for k, v in opt_split.items()},
            'model_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'best_val_loss': best_val_loss
        }
        all_fold_metrics.append(fold_result)
        all_losses.append(losses)
        
        # 保存模型和特征
        dataset_dir = f'results/{opt["dataset"]}_bestopt/fold_results'
        os.makedirs(dataset_dir, exist_ok=True)
        
        torch.save(best_val_model_state, f'{dataset_dir}/model_split{split_id}_time{opt["time"]}.pt')
        if best_val_features is not None:
            np.save(f'{dataset_dir}/val_features_split{split_id}_time{opt["time"]}.npy', best_val_features)
        if best_test_features is not None:
            np.save(f'{dataset_dir}/test_features_split{split_id}_time{opt["time"]}.npy', best_test_features)
    
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
    with open(f'{dataset_dir}/fold_metrics_time{opt["time"]}.json', 'w') as f:
        json.dump(all_fold_metrics, f, indent=2)
    with open(f'{dataset_dir}/summary_time{opt["time"]}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(f'{dataset_dir}/losses_time{opt["time"]}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'fold_{i}' for i in range(num_splits)])
        for row in zip(*all_losses):
            writer.writerow(row)
#   else:
#     main(opt)


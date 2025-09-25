from data_handling import get_data
import numpy as np
import torch.optim as optim
from models import *
from torch import nn
from best_params import best_params_dict
import torch
import argparse
import random

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
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed')

    args = parser.parse_args()
    cmd_opt = vars(args)

    opt = cmd_opt
    seed_all(opt['seed'])

    # best_opt = best_params_dict[cmd_opt['dataset']]
    # opt = {**cmd_opt, **best_opt}
    print(opt)

    n_splits = opt['n_splits']
    if n_splits == 1:
        best = train_GNN(opt,0)
        print('Test accuracy: ', best*100)
        exit()

    best = []
    for split in range(n_splits):
        best.append(train_GNN(opt,split))
    print('Mean test accuracy: ', np.mean(np.array(best)*100),'std: ', np.std(np.array(best)*100))


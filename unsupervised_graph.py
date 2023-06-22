# This is the usecase for graph classification task with minibatch training

import argparse
import numpy as np
import random
import os
import os.path as osp
import sys
sys.path.append('../')

import torch
from torch import nn
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn.inits import uniform
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import seed_everything
from Loss import JSD 
from Evaluator import get_split, LREvaluator
from ContrastMode import DualBranchContrast
from Augmentor import SpectralAugmentor

###################### Backbone Model ######################

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)

    
###################### GCL Encoder ######################

class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ptb_prob1 = data.max
        ptb_prob2 = data.min
        
        x1, edge_index1, _ = self.aug1(x, edge_index, ptb_prob1, batch)
        x2, edge_index2, _ = self.aug2(x, edge_index, ptb_prob2, batch)
        z1, g1 = self.gcn1(x1, edge_index1, batch)
        z2, g2 = self.gcn2(x2, edge_index2, batch)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2


###################### GCL Training and Testing ######################

def train(encoder_model, contrast_model, dataloader, optimizer, device):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        h1, h2, g1, g2 = encoder_model(data)
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def test(encoder_model, dataloader, device):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, _, g1, g2 = encoder_model(data)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # result = SVMEvaluator(linear=True)(x, y, split)
    
    best_result = {
        'accuracy': 0,
        'micro_f1': 0,
        'macro_f1': 0,
        'accuracy_val': 0,
        'micro_f1_val': 0,
        'macro_f1_val': 0
    }
    for decay in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
        result = LREvaluator(weight_decay=decay)(x, y, split)
        if result['accuracy_val'] > best_result['accuracy_val']:
            best_result = result
    return best_result

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--dataset', type=str, default='MUTAG') 
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--aug_lr1', type=float, default=0.5, help='augmentation learning rate for spectral max')
    parser.add_argument('--aug_lr2', type=float, default=0.5, help='augmentation learning rate for spectral min')
    parser.add_argument('--aug_iter', type=int, default=20, help='iteration for augmentation')
    parser.add_argument('--pf', type=float, default=0.4, help='feature masking probability')
    parser.add_argument('--pe', type=float, default=0.2, help='edge perturbation probability')
    return parser.parse_args()

def main():
    args = arg_parse()
    
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    # Load dataset
    path = osp.join(osp.expanduser('/data/lu/'), 'datasets')
    dataset = TUDataset(path, name=args.dataset)
    
    # Initialize augmentor
    aug1 = SpectralAugmentor(
        ratio=args.pe,
        lr=args.aug_lr1,
        iteration=args.aug_iter,
        dis_type='max',
        device=device
    )
    
    aug2 = SpectralAugmentor(
        ratio=args.pe,
        lr=args.aug_lr2,
        iteration=args.aug_iter,
        dis_type='min',
        device=device
    )
    
    # Pre-compute the perturbation probability or load the saved probability
    update_data_path = osp.join(path, args.dataset+'/updated_{}_{}_{}.pt'.format(args.aug_lr1, args.aug_lr2, args.pe))
    if os.path.exists(update_data_path):  # Load saved probability matrix
        update_data_ls = torch.load(update_data_path)
        print('(A): perturbation probability loaded!')
    else:  # Pre-compute perturbation probability matrix and save it to file
        print('(A): precomputing probability ...')
        assert dataset.len() > 1  # should have multiple data object for graph classification
        update_data_ls = []
        for i in tqdm(range(dataset.len())):
            data = dataset.get(i)        
            aug1.calc_prob(data, silence=True) # note that data is updated with data['max']=ptb_prob1
            aug2.calc_prob(data, silence=True) # note that data is further updated with data['min']=ptb_prob2
            update_data_ls.append(data)
        # Save the updated data to re-use in future
        torch.save(update_data_ls, update_data_path)
    
    # Start GCL training
    dataloader = DataLoader(update_data_ls, batch_size=32)  # batch size should be larger than 1
    gcn1 = GConv(input_dim=max(dataset.num_features, 1), hidden_dim=512, num_layers=2).to(device)
    gcn2 = GConv(input_dim=max(dataset.num_features, 1), hidden_dim=512, num_layers=2).to(device)
    mlp1 = FC(input_dim=512, output_dim=512)
    mlp2 = FC(input_dim=512 * 2, output_dim=512)
    encoder_model = Encoder(gcn1=gcn1, gcn2=gcn2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(device)
    contrast_model = DualBranchContrast(loss=JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.00001)
    with tqdm(total=args.epoch, desc='(T)') as pbar:
        for epoch in range(1, args.epoch+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader, device)
    print(f'(E): Test accuracy={test_result["accuracy"]:.4f}, F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

if __name__ == '__main__':
    main()

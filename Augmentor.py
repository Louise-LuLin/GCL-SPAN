###
# Modified based on PyGCL: https://github.com/PyGCL/PyGCL
###

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List

from tqdm import tqdm
import pickle as pkl
import os
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor

from torch_geometric.utils.sparse import to_edge_index
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Batch, Data
from utils import get_adj_tensor, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature


###################### Base Class ######################

class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    ptb_prob: Optional[SparseTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[SparseTensor]]:
        return self.x, self.edge_index, self.ptb_prob


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
        self, 
        x: torch.FloatTensor, 
        edge_index: torch.LongTensor, 
        ptb_prob: Optional[SparseTensor] = None, 
        batch = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(Graph(x, edge_index, ptb_prob), batch).unfold()
    
    
###################### Customized Class ######################

# compose multiple augmentors
class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g, batch)
        return g
    

# feature augmentor
class FeatureAugmentor(Augmentor):
    def __init__(self, pf: float):
        super(FeatureAugmentor, self).__init__()
        self.pf = pf

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        x, edge_index, _ = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, ptb_prob=None)

    def get_aug_name(self):
        return 'feature'
    
# spectral augmentor
class SpectralAugmentor(Augmentor):
    
    def __init__(self, ratio, lr, iteration, dis_type, device, sample='no', threshold=0.5):
        super(SpectralAugmentor, self).__init__()
        
        self.ratio = ratio
        self.lr = lr
        self.iteration = iteration
        self.dis_type = dis_type
        self.device = device
        self.sample = sample
        self.threshold = threshold
                
    def get_aug_name(self):
        return self.dis_type
    
    # precompute the perturbation propability based on spectral change
    def calc_prob(self, data, fast=False, check='no', save='no', verbose=False, silence=False):
        x, edge_index = data.x, data.edge_index
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # ori_adj = to_dense_adj(edge_index)
        
        nnodes = ori_adj.shape[0]
        adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(adj_changes, 1e-5, 1./nnodes)
        
        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        
        if fast: # only obtain k largest and/or smallest eigenvalues with faster eigendecomposition alg
            ori_e = torch.lobpcg(ori_adj_norm, k=10, largest=True)
        else:
            ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        eigen_norm = torch.norm(ori_e)
        
        n_perturbations = int(self.ratio * (ori_adj.sum()/2))
        with tqdm(total=self.iteration, desc='Spectral Augment-'+self.dis_type, disable=silence) as pbar:
            verb = max(1, int(self.iteration/10))
            for t in range(1, self.iteration+1):
                modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes))
                
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                # modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = get_normalize_adj_tensor(modified_adj_noise, device=self.device)
                
                if fast: # only obtain k largest and/or smallest eigenvalues with faster eigendecomposition alg
                    e = torch.lobpcg(adj_norm_noise, k=10, largest=True)
                else: 
                    e = torch.linalg.eigvalsh(adj_norm_noise)
                eigen_self = torch.norm(e)
                
                # spectral distance
                eigen_mse = torch.norm(ori_e-e)
                
                if self.dis_type == 'max-l2':
                    reg_loss = eigen_mse / eigen_norm
                elif self.dis_type == 'min-l2':
                    reg_loss = -eigen_mse / eigen_norm
                elif self.dis_type == 'max':
                    reg_loss = eigen_self / eigen_norm
                    
                    # n = 100
                    # idx = torch.argsort(e)[:n]
                    # mask = torch.zeros_like(e).bool()
                    # mask[idx] = True
                    # eigen_low = torch.norm(e*mask, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # idx2 = torch.argsort(e, descending=True)[:n]
                    # mask2 = torch.zeros_like(e).bool()
                    # mask2[idx2] = True
                    # eigen_high = torch.norm(e*mask2, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # reg_loss = eigen_low - eigen_high
                    
                elif self.dis_type == 'min':
                    reg_loss = -eigen_self / eigen_norm
                    
                    # n = 100
                    # idx = torch.argsort(e)[:n]
                    # mask = torch.zeros_like(e).bool()
                    # mask[idx] = True
                    # eigen_low = torch.norm(e*mask, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # idx2 = torch.argsort(e, descending=True)[:n]
                    # mask2 = torch.zeros_like(e).bool()
                    # mask2[idx2] = True
                    # eigen_high = torch.norm(e*mask2, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # reg_loss = - eigen_low + eigen_high
                    
                elif self.dis_type.startswith('max-low'):
                    # low-rank loss in GF-attack
                    n = int(self.dis_type.replace('max-low',''))
                    idx = torch.argsort(e)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = eigen_gf
                elif self.dis_type.startswith('min-low'):
                    # low-rank loss in GF-attack
                    n = int(self.dis_type.replace('min-low',''))
                    idx = torch.argsort(e)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = -eigen_gf
                elif self.dis_type.startswith('max-high'):
                    # high-rank loss in GF-attack
                    n = int(self.dis_type.replace('max-high',''))
                    idx = torch.argsort(e, descending=True)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = eigen_gf
                elif self.dis_type.startswith('min-high'):
                    # high-rank loss in GF-attack
                    n = int(self.dis_type.replace('min-high',''))
                    idx = torch.argsort(e, descending=True)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = -eigen_gf
                elif self.dis_type.startswith('sep'):
                    
                    # n = int(self.dis_type.replace('sep',''))
                    # mask = torch.zeros_like(e).bool()
                    # mask[-n:] = True
                    # eigen_high = torch.masked_select(e, mask)
                    # reg_loss = eigen_high @ eigen_high.t() / (e @ e.t())
                    
                    # mask = e.ge(0.0)  # [-1, 1]
                    # eigen_high = torch.masked_select(e, mask)
                    # reg_loss = eigen_high @ eigen_high.t() / (e @ e.t())
                    
                    mask = e.le(0.0)  # [-1, 1]
                    ori_high = torch.masked_select(ori_e, mask)
                    high = torch.masked_select(e, mask)
                    mask2 = e.ge(0.0)
                    ori_low = torch.masked_select(ori_e, mask2)
                    low = torch.masked_select(e, mask2)
                    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    # reg_loss = cos(ori_high, high) + 1 - cos(ori_low, low)
                    reg_loss = cos(ori_high, high)
                    # reg_loss = torch.norm(ori_high - high, p=2)**2 / torch.norm(ori_e - e, p=2)**2
                else:
                    exit(f'unknown distance metric: {self.dis_type}')
                
                self.loss = reg_loss
                
                adj_grad = torch.autograd.grad(self.loss, adj_changes)[0]

                lr = self.lr / np.sqrt(t+1)
                adj_changes.data.add_(lr * adj_grad)
                
                before_p = torch.clamp(adj_changes, 0, 1).sum()
                before_l = adj_changes.min()
                before_r = adj_changes.max()
                before_m = torch.clamp(adj_changes, 0, 1).sum()/torch.count_nonzero(adj_changes)
                self.projection(n_perturbations, adj_changes)
                after_p = adj_changes.sum()
                after_l = adj_changes.min()
                after_r = adj_changes.max()
                after_m = adj_changes.sum()/torch.count_nonzero(adj_changes)
                
                if verbose and t%verb == 0:
                    print (
                        '-- Epoch {}, '.format(t), 
                        'reg loss = {:.4f} | '.format(reg_loss),
                        'ptb budget/b/a = {:.1f}/{:.1f}/{:.1f}'.format(n_perturbations, before_p, after_p),
                        'min b/a = {:.4f}/{:.4f}'.format(before_l, after_l),
                        'max b/a = {:.4f}/{:.4f}'.format(before_r, after_r),
                        'mean b/a = {:.4f}/{:.4f}'.format(before_m, after_m))
                    self.check_hist()

                pbar.set_postfix({'reg_loss': reg_loss.item(), 'budget': n_perturbations, 'b_proj': before_p.item(), 'a_proj': after_p.item()})
                pbar.update()
    
        data[self.dis_type] = SparseTensor.from_dense(self.reshape_m(nnodes, adj_changes))
    
#         if check == 'yes':
#             self.check_changes(ori_adj, adj_changes, y)
            
#         if save == 'yes':
#             out_dir = '../check'
#             os.makedirs(out_dir, exist_ok=True)
            
#             output_path = os.path.join(out_dir, self.dis_type+'_'+str(self.ratio)+'_'+str(self.lr)+'_'+str(self.iteration)+'.bin')
#             res = {'ori_e': ori_e, 'e': e, 'adj_change': adj_changes.detach().cpu(), 'ori_adj': ori_adj.detach().cpu()}
#             with open(output_path, 'wb') as file:
#                 pkl.dump(res, file)
        
        return data
                
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        x, edge_index, ptb_prob = g.unfold()

        ori_adj = to_dense_adj(edge_index, batch) 
        ptb_idx, ptb_w = to_edge_index(ptb_prob)
        ptb_m = to_dense_adj(ptb_idx, batch, ptb_w)
               
        ptb_adj = self.random_sample(ptb_m)
        
        modified_adj = self.get_modified_adj(ori_adj, ptb_adj).detach()
        
        self.check_adj_tensor(modified_adj)
        
        if batch is None: # full batch training
            edge_index, _ = dense_to_sparse(modified_adj)
        else:  # minibatch training
            # edge_index, _ = dense_to_sparse(modified_adj) # Wrong! 
            x_unbatched = unbatch(x, batch)
            aug_data = Batch.from_data_list([Data(x=x_unbatched[b], edge_index=dense_to_sparse(modified_adj[b])[0]) for b in range(modified_adj.shape[0])])
            x = aug_data.x
            edge_index = aug_data.edge_index
        
        return Graph(x=x, edge_index=edge_index, ptb_prob=None)
    

    def get_modified_adj(self, ori_adj, m):
        nnodes = ori_adj.shape[1]
        complementary = (torch.ones_like(ori_adj) - torch.eye(nnodes).to(self.device) - ori_adj) - ori_adj
        modified_adj = complementary * m + ori_adj
        
        return modified_adj
    
    def reshape_m(self, nnodes, adj_changes):
        m = torch.zeros((nnodes, nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()
        return m
    
    def add_random_noise(self, ori_adj):
        nnodes = ori_adj.shape[0]
        noise = 1e-4 * torch.rand(nnodes, nnodes).to(self.device)
        return (noise + torch.transpose(noise, 0, 1))/2.0 + ori_adj
    
    def projection(self, n_perturbations, adj_changes):
        if torch.clamp(adj_changes, 0, self.threshold).sum() > n_perturbations:
            left = (adj_changes).min()
            right = adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, 1e-4, adj_changes)
            l = left.cpu().detach()
            r = right.cpu().detach()
            m = miu.cpu().detach()
            adj_changes.data.copy_(torch.clamp(adj_changes.data-miu, min=0, max=1))
        else:
            adj_changes.data.copy_(torch.clamp(adj_changes.data, min=0, max=1))
            
            
    def bisection(self, a, b, n_perturbations, epsilon, adj_changes):
        def func(x):
            return torch.clamp(adj_changes-x, 0, self.threshold).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                b = miu
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    
    def random_sample(self, edge_prop):
        with torch.no_grad():
            s = edge_prop.cpu().detach().numpy()
            # s = (s + np.transpose(s))
            if self.sample == 'yes':
                binary = np.random.binomial(1, s)
                mask = np.random.binomial(1, 0.7, s.shape)
                sampled = np.multiply(binary, mask)
            else:
                sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
    
    
    #############################################################
    # check intermediate results
    
    
    def check_hist(self, adj_changes):
        with torch.no_grad():
            s = adj_changes.cpu().detach().numpy()
            stat = {}
            stat['1.0'] = (s==1.0).sum()
            stat['(1.0,0.8)'] = (s>0.8).sum() - (s==1.0).sum()
            stat['[0.8,0.6)'] = (s>0.6).sum() - (s>0.8).sum()
            stat['[0.6,0.4)'] = (s>0.4).sum() - (s>0.6).sum()
            stat['[0.4,0.2)'] = (s>0.2).sum() - (s>0.4).sum()
            stat['[0.2,0.0]'] = (s>0.0).sum() - (s>0.2).sum()
            stat['0.0'] = (s==0.0).sum()
            print (stat)
    
    
    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is unweighted, all-zero diagonal.
        """
        # assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj[0].diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"
        
        
    def check_changes(self, ori_adj, adj_changes, y):
        nnodes = ori_adj.shape[0]
        
        m = torch.zeros((nnodes, nnodes))
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes.cpu()
        m = m + m.t()
        idx = torch.nonzero(m).numpy()
        m = m.detach().numpy()
        degree = ori_adj.sum(dim=1).cpu().numpy()
        idx2 = torch.nonzero(ori_adj.cpu()).numpy()
        
        stat = {'intra': 0, 'inter': 0, 'degree': [], 'inter_add':0, 'inter_rm': 0, 'intra_add': 0, 'intra_rm': 0, 'degree_add': [], 'degree_rm': []}
        for i in tqdm(idx):
            d = degree[i[0]] + degree[i[1]]
            if ori_adj[i[0], i[1]] == 1:  # rm
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_rm'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_rm'] += m[i[0], i[1]]
                stat['degree_rm'].append(d/2)
            if ori_adj[i[0], i[1]] == 0:  # add
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_add'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_add'] += m[i[0], i[1]]
                stat['degree_add'].append(d/2)
        for i in tqdm(idx2):
            d = degree[i[0]] + degree[i[1]]
            if y[i[0]] == y[i[1]]:  # intra
                stat['intra'] += 1
            if y[i[0]] != y[i[1]]:  # inter
                stat['inter'] += 1
            stat['degree'].append(d/2)
                
        stat['degree_rm'] = sum(stat['degree_rm'])/(len(stat['degree_rm'])+0.1)
        stat['degree_add'] = sum(stat['degree_add'])/(len(stat['degree_add'])+0.1)
        stat['degree'] = sum(stat['degree'])/(len(stat['degree'])+0.1)
        
        print(stat)

    def augment_on_the_fly(self, g: Graph) -> Graph:
        x, edge_index, edge_prob = g.unfold()
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # ori_adj = to_dense_adj(edge_index)
        
        nnodes = ori_adj.shape[0]
                
        adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(adj_changes, 0.0, 0.001)
        
        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        eigen_norm = torch.norm(ori_e)
        
        # print(ori_adj.shape, ori_adj_norm.shape)
        # exit('')
        
        n_perturbations = int(self.ratio * (ori_adj.sum()/2))
        with tqdm(total=self.iteration, desc='Spectral Augment') as pbar:
            for t in range(1, self.iteration+1):
                modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes))
                
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = get_normalize_adj_tensor(modified_adj_noise, device=self.device)
                # e = torch.linalg.eigvalsh(adj_norm_noise)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_self = torch.norm(e)
                
                # spectral distance
                eigen_mse = torch.norm(ori_e-e)
                
                if self.dis_type == 'l2':
                    reg_loss = eigen_mse / eigen_norm
                elif self.dis_type == 'normDiv':
                    reg_loss = eigen_self / eigen_norm
                else:
                    exit(f'unknown distance metric: {self.dis_type}')

                self.loss = reg_loss
                
                adj_grad = torch.autograd.grad(self.loss, adj_changes)[0]

                lr = self.lr / np.sqrt(t+1)
                adj_changes.data.add_(lr * adj_grad)
                    
                before_p = torch.clamp(adj_changes, 0, 1).sum()
                self.projection(n_perturbations, adj_changes)
                after_p = torch.clamp(adj_changes, 0, 1).sum()
                
                pbar.set_postfix({'reg_loss': reg_loss.item(), 'eigen_mse': eigen_mse.item(), 'before_p': before_p.item(), 'after_p': after_p.item()})
                pbar.update()
        
        adj_changes = self.random_sample(adj_changes)
        
        modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes)).detach()
        self.check_adj_tensor(modified_adj)
            
        edge_index, _ = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index)

    
    

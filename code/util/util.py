import h5py, math, os, torch
import pandas as pd
import numpy as np
import cv2
from Bio import SeqIO
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sklearn import metrics
from scipy import stats

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

amino_acids = ['PAD','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def smiles2onehot(pdbid):
    seq = pd.read_csv('train_set/drug_smiles/' + pdbid + '.smi', header=None).to_numpy().tolist()[0][0].split('\t')[0]
    integer_encoder = []
    onehot_encoder = []
    for item in seq:
        integer_encoder.append(smiles_dict[item])
    for index in integer_encoder:
        temp = [0 for _ in range(len(smiles_dict) + 1)]
        temp[index] = 1
        onehot_encoder.append(temp)
    return onehot_encoder

def protein2onehot(pdbid):
    for seq_recoder in SeqIO.parse('train_set/target_fasta/' + pdbid + '.fasta', 'fasta'):
        seq = seq_recoder.seq
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def _to_onehot(data, max_len):
    feature_list = []
    for seq in data:
        if max_len == 1000:
            feature = protein2onehot(seq)
            if len(feature) > 1000:
                feature = feature[:1000]
            feature_list.append(feature)
        elif max_len == 150:
            feature = smiles2onehot(seq)
            if len(feature) > 150:
                feature = feature[:150]
            feature_list.append(feature)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 1000:
                    temp = [0] * 21
                    temp[0] = 1
                elif max_len == 150:
                    temp = [0] * 65
                    temp[0] = 1
                feature_list[i].append(temp)
    return torch.from_numpy(np.array(feature_list, dtype=np.float32))

def img_resize(data):
    data_list = []
    for id in data:
        img = np.load('train_set/distance_matrix/' + id + '.npz')['map']
        img_resize = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        data_list.append(img_resize)
    return np.array(data_list)

class CompoundDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None , compound=None, protein=None, affinity=None, transform=None, pre_transform=None, compound_graph=None, protein_graph=None):
        super(CompoundDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(compound, affinity, compound_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self, compound, affinity, compound_graph):
        assert (len(compound) == len(affinity)), '这两个列表必须是相同的长度!'
        data_list = []
        data_len = len(compound)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len)) # Convert molecular format to graphical structure
            smiles = compound[i]
            # target = protein[i]
            label = affinity[i]
            print(smiles)
            # print(target)
            print(label)

            size, features, edge_index = compound_graph[i][smiles]
            GCNCompound = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.FloatTensor([label]), id=smiles)
            GCNCompound.__setitem__('size', torch.LongTensor([size]))
            data_list.append(GCNCompound)
            # data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CMD(nn.Module):

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        # 计算均值
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        
        # 计算中心化向量
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        
        # 对中心化向量进行标准化处理
        sx1_std = torch.std(sx1, dim=0) + 1e-8
        sx2_std = torch.std(sx2, dim=0) + 1e-8
        sx1 = sx1 / sx1_std
        sx2 = sx2 / sx2_std

        # 计算均值差异
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

def moe_nig(*dist_params):
    """
    多模态NIG融合函数
    通过加权平均将不同模态的输出进行融合
    """
    if len(dist_params) % 4 != 0:
        raise ValueError("输入参数必须是4的倍数")
    
    num_dist = len(dist_params) // 4
    if num_dist == 1:
        return dist_params  # 单个分布直接返回
    
    # 将参数分组为分布元组
    distributions = [
        dist_params[i*4:(i+1)*4] 
        for i in range(num_dist)
    ]
    
    # 初始化为第一个分布
    u, la, alpha, beta = distributions[0]
    
    # 逐步融合后续分布
    for i in range(1, num_dist):
        u_i, la_i, alpha_i, beta_i = distributions[i]
        
        # Eq. 9: 正态逆伽玛分布的融合公式
        u_new = (la * u + la_i * u_i) / (la + la_i)
        la_new = la + la_i
        alpha_new = alpha + alpha_i + 0.5
        beta_new = beta + beta_i + 0.5 * (la * (u - u_new) ** 2 + la_i * (u_i - u_new) ** 2)
        
        u, la, alpha, beta = u_new, la_new, alpha_new, beta_new
    
    return u, la, alpha, beta

def criterion_nig(u, la, alpha, beta, y):
    """
    正态逆伽玛分布的损失函数
    包含回归损失和正则化项
    """
    # 基础损失计算
    om = 2 * beta * (1 + la)
    loss = sum(0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)) / len(u)
    # 正则化项
    lossr = 0.01 * sum(torch.abs(u - y) * (2 * la + alpha)) / len(u)
    loss = loss + lossr
    return loss

def mae(y, f):
    mae = metrics.mean_absolute_error(y, f)
    return mae

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def pearson(y, f):
    rp = stats.pearsonr(y, f)[0]
    return rp

def r_squared(y, f):
    sse = np.sum((y - f) ** 2)
    ssr = np.sum((f - np.mean(y)) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    # r2 = metrics.r2_score(y, f)
    return r2

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
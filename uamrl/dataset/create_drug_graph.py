import os
import numpy as np
import pandas as pd

from uamrl.util.graphUtil import getCompoundGraph
from uamrl.util.util import *
from tqdm import tqdm

train_id = pd.read_csv('data/train_data.csv')['PDBID'].to_numpy().tolist()
val_id = pd.read_csv('data/val_data.csv')['PDBID'].to_numpy().tolist()
test_id = pd.read_csv('data/test_data.csv')['PDBID'].to_numpy().tolist()

train_affinity = pd.read_csv('data/train_data.csv')['affinity'].to_numpy().tolist()
val_affinity = pd.read_csv('data/val_data.csv')['affinity'].to_numpy().tolist()
test_affinity = pd.read_csv('data/test_data.csv')['affinity'].to_numpy().tolist()

if __name__ == '__main__':

    test_drug_graph = []
    for i in tqdm(range(len(test_id))):
        drug_info = {}
        g = getCompoundGraph(test_id[i])
        drug_info[test_id[i]] = g
        test_drug_graph.append(drug_info)
    print('测试集药物转换完成')

    test_id, test_affinity =  np.asarray(test_id), np.asarray(test_affinity)
    print('准备将药物测试集数据转化为Pytorch格式')
    protein_train_data = CompoundDataset(root='train_set', dataset='test_data', compound=test_id, compound_graph=test_drug_graph, affinity=test_affinity)
    print('药物测试集集数据转化为Pytorch格式完成')


    train_drug_graph = []
    for i in tqdm(range(len(train_id))):
        drug_info = {}
        g = getCompoundGraph(train_id[i])
        drug_info[train_id[i]] = g
        train_drug_graph.append(drug_info)
    print('训练集药物转换完成')

    train_id, train_affinity =  np.asarray(train_id), np.asarray(train_affinity)
    print('准备将药物测试集数据转化为Pytorch格式')
    protein_train_data = CompoundDataset(root='train_set', dataset='train_data', compound=train_id, compound_graph=train_drug_graph, affinity=train_affinity)
    print('药物测试集集数据转化为Pytorch格式完成')

    train_drug_graph = []
    for i in tqdm(range(len(val_id))):
        drug_info = {}
        g = getCompoundGraph(val_id[i])
        drug_info[val_id[i]] = g
        train_drug_graph.append(drug_info)
    print('训练集药物转换完成')

    val_id, val_affinity =  np.asarray(val_id), np.asarray(val_affinity)
    print('准备将药物测试集数据转化为Pytorch格式')
    protein_train_data = CompoundDataset(root='train_set', dataset='val_data', compound=val_id, compound_graph=train_drug_graph, affinity=val_affinity)
    print('药物测试集集数据转化为Pytorch格式完成')



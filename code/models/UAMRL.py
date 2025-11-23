import os, sys
import numpy as np
import torch
import torch.nn as nn
from util.util import *
from torch_geometric.nn import SAGEConv, global_add_pool as gap
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch.nn.functional as F
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import esm as ESM

device = torch.device('cuda')  

class Sequence_Model(nn.Module):
    def __init__(self, in_channel, embedding_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1, relative_position=False, Heads=None, use_residue=False):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        self.residue_in_channel = 64
        self.dim = '1d'
        self.dropout = 0.1
        self.relative_position = relative_position
        self.use_residue = use_residue
        
        self.emb = nn.Linear(in_channel, embedding_channel)
        self.dropout = nn.Dropout(self.dropout)
        
        self.layers = nn.Sequential(
            nn.Conv1d(embedding_channel, med_channel[1], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[1], med_channel[2], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[2], out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x = self.dropout(self.emb(x))
        x = self.layers(x.permute(0, 2, 1)).view(-1, 256)
        return x
       
class Flat_Model(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Flat_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, med_channel[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[1], med_channel[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[2], out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.AdaptiveMaxPool2d(1)
        )
        
    def forward(self, x):
        x = self.layers(x).view(-1, 256)
        return x

class GraphConv(nn.Module):
    def __init__(self, feature_dim, emb_dim, hidden_dim=32, output_dim=256, dropout=0.1):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.emb = nn.Linear(feature_dim, emb_dim)
        self.cconv1 = SAGEConv(emb_dim, hidden_dim, aggr='sum')
        self.cconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
        self.cconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.flat = nn.Linear(hidden_dim * 4, output_dim)

    
    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        compound_feature, compound_index, compound_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        compound_feature = self.dropout(self.emb(compound_feature))

        compound_feature = self.cconv1(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv2(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv3(compound_feature, compound_index)

        # 对卷积后的小分子进行图的池化操作
        compound_feature = gap(compound_feature, compound_batch)

        compound_feature = self.flat(compound_feature)

        return compound_feature

class Prot_Bert(nn.Module):
    def __init__(self, ):
        super(Prot_Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        self.model = BertModel.from_pretrained('Rostlab/prot_bert')
        self.output_layer = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU())

    def forward(self, data):
        feature_list = torch.Tensor().to(device)
        for data_id in data.id:
            for seq_recoder in SeqIO.parse(f'{rootPath}/train_set/target_fasta/{data_id}.fasta', 'fasta'):
                seq = seq_recoder.seq
            integer_encoded = str(seq)
            inputs = self.tokenizer(integer_encoded, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            feature = outputs.pooler_output          
            feature_list =  torch.cat((feature_list, feature), dim = 0)
        output = self.output_layer(feature_list)
        return output       

class Comp_Bert(nn.Module):
    def __init__(self, ):
        super(Comp_Bert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', do_lower_case=False)
        self.model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.output_layer = nn.Sequential(nn.Linear(768, 256), nn.LeakyReLU())

    def forward(self, data):
        feature_list = torch.Tensor().to(device)
        for data_id in data.id:
            seq = pd.read_csv(f'{rootPath}/train_set/drug_smiles/{data_id}.smi', header=None).to_numpy().tolist()[0][0].split('\t')[0]
            inputs = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                feature = outputs.pooler_output
            feature_list =  torch.cat((feature_list, feature), dim = 0)
        output = self.output_layer(feature_list)
        return output

class ESM_DBP(nn.Module):
    def __init__(self, ):
        super(ESM_DBP, self).__init__()
        esm = ESM.ESM2()
        self.esm_model = torch.nn.DataParallel(esm)
        self.esm_model.load_state_dict(torch.load(f'{curPath}/ESM-DBP.model', map_location=lambda storage, loc: storage, weights_only=True))
        self.alphabet = ESM.data.Alphabet.from_architecture("ESM-1b")

    def get_one_prediction_res(self, batch_tokens):
        results = self.esm_model(tokens=batch_tokens, repr_layers=[33], return_contacts=False) # logits: 1, num + 2, 33; representations: 33: 1, num + 2, 1280
        token_representations = torch.squeeze(results["representations"][33]) # 1, num + 2, 1280
        fea_represent = token_representations[1:-1] # num, 1280
        c_fea_represent=torch.mean(fea_represent,dim=0).unsqueeze(dim=0) # 1, 1280
        return c_fea_represent

    def forward(self, data):
        feature_list = torch.tensor([]).reshape(0, 1280).to(device)
        for data_id in data.id:
            for seq_recoder in SeqIO.parse(f'{rootPath}/train_set/target_fasta/{data_id}.fasta', 'fasta'):
                seq = seq_recoder.seq
                datazip = [(data_id, seq)]
                batch_converter = self.alphabet.get_batch_converter()
                batch_labels, batch_strs, batch_tokens = batch_converter(datazip) # 1, num + 2
                feature = self.get_one_prediction_res(batch_tokens)
                feature_list = torch.cat((feature_list, feature), dim = 0)
        return feature_list   

class UAMRL(nn.Module):
    def __init__(self, compound_sequence_channel, protein_sequence_channel, med_channel, hidden_size, embedding_dim = 128, output_size=1):
        super(UAMRL, self).__init__()
        
        self.compound_sequence = Sequence_Model(compound_sequence_channel, embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1)
        # self.protein_sequence = Sequence_Model(protein_sequence_channel, embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1)
        self.compound_stru = GraphConv(27, embedding_dim)
        self.protein_stru = Flat_Model(1, med_channel, hidden_size)
        # self.compound_sequence = Comp_Bert()
        self.protein_sequence = Prot_Bert()

        # ESM 编码器
        # self.esm = ESM_DBP()
        # self.protein_sequence_esm = nn.Sequential(nn.Linear(1280, 1000), nn.BatchNorm1d(1000), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(500, hidden_size))

        # 共享层
        self.shared_layer = self._create_shared_private_layer(512, 512)
        
        # 私有层
        self.private_layers = nn.ModuleDict({
            mode: self._create_shared_private_layer(512, 512)
            for mode in ['ss', 'gg', 'sg', 'gs']
        })

        # NIG参数预测
        self.shared_outputs = nn.ModuleDict()
        self.private_outputs = nn.ModuleDict()
        for mode in ['ss', 'gg', 'sg', 'gs']:
            for param in ['mu', 'v', 'alpha', 'beta']:
                self.shared_outputs[f"{mode}_{param}"] = self._create_output_layer(512, output_size)
                self.private_outputs[f"{mode}_{param}"] = self._create_output_layer(512, output_size)

        self.loss_cmd = CMD()

    def _create_shared_private_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim)
        )
    
    def _create_output_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim)
        )
     
    def evidence(self, x):
        # 用于将输入通过 softplus 函数激活，确保输出为正值。这用于计算 NIG 分布中的参数，如方差、alpha、beta等。
        return F.softplus(x)

    def split(self, mu, logv, logalpha, logbeta):
        # 用于将预测的输出分解为 mu（均值）、v（方差）、alpha 和 beta，并通过 evidence() 函数保证它们为正值。
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta
        
    def get_cmdloss(self, x1, x2):
        return self.loss_cmd(x1, x2, 5)

    def forward(self, compound_sequence, protein_sequence, protein_graph, data):
        
        # 特征提取
        c_seq = self.compound_sequence(compound_sequence)
        c_graph = self.compound_stru(data)
        p_seq = self.protein_sequence(data)        
        # ESM 使用
        # with torch.no_grad():
        #     protein_sequence = self.esm(data)
        #     p_seq = self.protein_sequence(protein_sequence) 
        p_graph = self.protein_stru(protein_graph)

        # 特征组合
        features = {
            'ss': torch.cat((c_seq, p_seq), dim=1),
            'gg': torch.cat((c_graph, p_graph), dim=1),
            'sg': torch.cat((c_seq, p_graph), dim=1),
            'gs': torch.cat((c_graph, p_seq), dim=1)
        }

        # 共享特征处理
        shared_features = {mode: self.shared_layer(feat) for mode, feat in features.items()}

        # 私有特征处理
        private_features = {mode: self.private_layers[mode](feat) for mode, feat in features.items()}

        # 计算CMD损失
        cmd_loss = sum(
            self.get_cmdloss(shared_features[m1], shared_features[m2])
            for i, m1 in enumerate(shared_features)
            for j, m2 in enumerate(shared_features)
            if i < j
        ) / 6

        # 结果处理函数
        def process_outputs(feature_dict, output_dict):
            results = {}
            for mode in feature_dict:
                params = {}
                for param in ['mu', 'v', 'alpha', 'beta']:
                    net = output_dict[f"{mode}_{param}"]
                    params[param] = net(feature_dict[mode])
                
                results[mode] = self.split(
                    params['mu'], params['v'], 
                    params['alpha'], params['beta']
                )
            return results
        
        # 处理共享和私有输出
        shared_results = process_outputs(shared_features, self.shared_outputs)
        private_results = process_outputs(private_features, self.private_outputs)

        return (
            *shared_results['ss'], *shared_results['gg'],
            *shared_results['sg'], *shared_results['gs'],
            *private_results['ss'], *private_results['gg'],
            *private_results['sg'], *private_results['gs']
        ),cmd_loss

 
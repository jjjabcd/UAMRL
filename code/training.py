import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util.util import *
from util.util import _to_onehot
from config import get_config
from models.UAMRL import UAMRL
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluate_metrics import *
import prettytable as pt
import networkx as nx 
import matplotlib.pyplot as plt
device = torch.device('cuda:0')

def training(model, train_loader, optimizer, config, epoch, epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red', ncols=100)
    training_loss = 0.0
    for batch, data in loop:
        if len(data) < 2:
            break
        compound_sequence = _to_onehot(data.id, 150).to(device)
        protein_sequence = _to_onehot(data.id, 1000).to(device)
        protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
        model_output, cmd_loss = model(compound_sequence, protein_sequence, protein_img, data.to(device))
     
        mu, v, alpha, beta = moe_nig(model_output)
        
        (mu_ss, v_ss, alpha_ss, beta_ss,
         mu_gg, v_gg, alpha_gg, beta_gg,
         mu_sg, v_sg, alpha_sg, beta_sg,
         mu_gs, v_gs, alpha_gs, beta_gs,
         mu_ss_private, v_ss_private, alpha_ss_private, beta_ss_private,
         mu_gg_private, v_gg_private, alpha_gg_private, beta_gg_private,
         mu_sg_private, v_sg_private, alpha_sg_private, beta_sg_private,
         mu_gs_private, v_gs_private, alpha_gs_private, beta_gs_private) = model_output
        
        targets = data.y.view(-1, 1).to(torch.float).to(device)
        raw_loss = criterion_nig(mu_ss, v_ss, alpha_ss, beta_ss, targets) + \
                   criterion_nig(mu_gg, v_gg, alpha_gg, beta_gg, targets) + \
                   criterion_nig(mu_sg, v_sg, alpha_sg, beta_sg, targets) + \
                   criterion_nig(mu_gs, v_gs, alpha_gs, beta_gs, targets) + \
                   criterion_nig(mu_ss_private, v_ss_private, alpha_ss_private, beta_ss_private, targets) + \
                   criterion_nig(mu_gg_private, v_gg_private, alpha_gg_private, beta_gg_private, targets) + \
                   criterion_nig(mu_sg_private, v_sg_private, alpha_sg_private, beta_sg_private, targets) + \
                   criterion_nig(mu_gs_private, v_gs_private, alpha_gs_private, beta_gs_private, targets) + \
                   criterion_nig(mu, v, alpha, beta, targets)
        combined_loss = raw_loss  + cmd_loss*config.cmd_weight

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        training_loss += combined_loss.item()

        loop.set_postfix(loss=raw_loss.item(),cmdloss=cmd_loss.item())

    print(f"cls loss:{np.mean(training_loss):.4f}")

def validation(model, loader, config, epoch=1, epochs=1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), colour='blue', ncols=75)
        
        for batch, data in loop:
            compound_sequence = _to_onehot(data.id, 150).to(device)
            protein_sequence = _to_onehot(data.id, 1000).to(device)
            protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
            model_output, = model(compound_sequence, protein_sequence, protein_img,  data.to(device))
            output, = moe_nig(model_output)

            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds

if __name__ == '__main__':

    config = get_config()

    # load dataset
    batch_size = config.batch_size
    train_data = CompoundDataset(root='train_set', dataset='train_data')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CompoundDataset(root='train_set', dataset='test_data')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_data = CompoundDataset(root='train_set', dataset='val_data')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    epochs = config.epochs
    compound_sequence_dim = config.compound_sequence_dim
    protein_sequence_dim = config.protein_sequence_dim
    learning_rate = config.learning_rate
    patience = config.patience
    curr_patience = patience

    model = UAMRL(compound_sequence_dim, protein_sequence_dim, [32, 64, 128] , 256).to(device)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):  # 检查是否为线性层
            torch.nn.init.xavier_normal_(layer.weight)  # 初始化权重
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)  # 初始化偏置为零

    best_result = 10.0
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for epoch in range(1, epochs + 1):

        training(model, train_loader, optimizer, config, epoch, epochs)

        val_labels, val_preds = validation(model, val_loader, config, epoch, epochs)
        val_result = [mae(val_labels, val_preds), rmse(val_labels, val_preds), pearson(val_labels, val_preds)]
        
        test_labels, test_preds = validation(model, test_loader, config, epoch, epochs)
        test_result = [mae(test_labels, test_preds), rmse(test_labels, test_preds), pearson(test_labels, test_preds), ci(test_labels, test_preds), r_squared(test_labels, test_preds)]

        tb = pt.PrettyTable()
        tb.field_names = ['Epoch / curr_p ', 'Set', 'MAE', 'RMSE', 'PCC', 'CI', 'R-Squared']
        tb.add_row([f'{epoch} / {curr_patience}', 'Validation', f'{test_result[0]:.4f}', f'{test_result[1]:.4f}', f'{test_result[2]:.4f}','',''])
        tb.add_row([f'{epoch} / {curr_patience}', 'Test', f'{test_result[0]:.4f}', f'{test_result[1]:.4f}', f'{test_result[2]:.4f}', f'{test_result[3]:.4f}', f'{test_result[4]:.4f}'])

        print(tb)
        with open(f'result/{config.name}.txt', 'a') as write:
            write.writelines(str(tb) + '\n')
        if test_result[1] < best_result:
            best_result = test_result[1]
            curr_patience = patience
            print("Found new best model on dev set!")
            # 修改优化器的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate  # 修改学习率
            print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
            torch.save(model.state_dict(), f'data/best_model/model_{config.name}.pt')
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                curr_patience = patience
                scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    

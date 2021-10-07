import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gen import GEN
from models.hgc_gcn import HGC_GCN
from utils import *
from torch_geometric.utils import true_positive, false_positive, false_negative, true_negative
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

torch.cuda.empty_cache()

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
    # return total_labels.numpy(), total_preds.numpy()

# def evaluate(model, loader):
#     model.eval()
#     predictions = torch.Tensor()
#     affinity = torch.Tensor()
#
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             pred = model(data).detach().cpu().numpy()
#             label = data.y.cpu().numpy()
#             predictions.append(pred)
#             affinity.append(label)
#
#     predictions = np.concatenate(predictions)
#     affinity = np.concatenate(affinity)
#
#     fpr, tpr, thresholds = roc_curve(affinity[:,1], predictions[:,1], pos_label=2)
#
#     return roc_auc_score(affinity[:,1], predictions[:,1]), auc(fpr,tpr)

datasets = [['davis', 'kiba', 'allergy'][int(sys.argv[1])]]
modeling = [GEN, HGC_GCN][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run prepare_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        result_full = 'result_full_' + model_st + '_' + dataset + '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            G, P = predicting(model, device, test_loader)
            re = [epoch, mse(G, P), ci(G, P)]
            with open(result_full, 'a') as f:
                f.writelines(','.join(map(str, re)) + '\n')
        # for epoch in range(NUM_EPOCHS):
        #     train(model, device, train_loader, optimizer, epoch + 1)
        #     G, P = predicting(model, device, test_loader)
        #     # ROC_AUC, AUC = evaluate(model, test_loader)
        #     ret = [epoch, rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        #     if ret[1] < best_mse:
        #         torch.save(model.state_dict(), model_file_name)
        #         with open(result_file_name, 'a') as f:
        #             f.writelines(','.join(map(str, ret))+'\n')
        #         best_epoch = epoch + 1
        #         best_mse = ret[1]
        #         best_ci = ret[-1]
        #
        #         print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
        #     else:
        #         print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
        #               model_st, dataset)
            # x.append(epoch)
            # y.append(mse(G, P))
            # plt.plot(x, y)
            # plt.grid(True)
            # plt.xlabel('epoch')
            # plt.ylabel('mse')
            # plt.savefig(model_st + '_' + dataset + '_mse.png')

            # x = best_epoch
            # y = best_mse



        # G, P = predicting(model, device, test_loader)
        # x = FP_Rate(G, P)
        # y = TP_Rate(G, P)
        # plt.plot(x, y)
        # plt.xlabel('FP Rate')
        # plt.ylabel('TP Rate')
        # plt.title('ROC')
        # plt.savefig()
                # create ROC curves
            # y_test = np.concatenate(G)
            # y_predict = np.concatenate(P)
            # fpr_gnn, tpr_gnn, threshold_gnn = roc_curve(y_test[:, 1], y_predict[:, 1])
            #
            #     # plot ROC curves
            # plt.figure()
            # plt.plot(tpr_gnn, fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn, tpr_gnn) * 100))
            # plt.xlabel(r'True positive rate')
            # plt.ylabel(r'False positive rate')
            # plt.semilogy()
            # plt.ylim(0.001, 1)
            # plt.xlim(0, 1)
            # plt.grid(True)
            # plt.legend(loc='upper left')
            # plt.show()

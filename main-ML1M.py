import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from GOSR-ML1M import GOSR, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder, data_process
from utils2 import indicators_5, slid_window
import warnings
import argparse
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger

np.set_printoptions(threshold=np.inf)

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Games', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--user_update', default='rnn')
parser.add_argument('--item_update', default='rnn')
parser.add_argument('--user_long', default='orgat')
parser.add_argument('--item_long', default='orgat')
parser.add_argument('--user_short', default='att')
parser.add_argument('--item_short', default='att')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=3, help='sub-graph size')
parser.add_argument('--gpu', default='4')
parser.add_argument('--last_item', action='store_true', help='aggreate last item')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--model_record", action='store_true', default=False, help='record model')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda')
print(opt)

if opt.record:
    log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
               f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
               f'_layer_{opt.layer_num}_l2_{opt.l2}'
    mkdir_if_not_exist(log_file)
    sys.stdout = Logger(log_file)
    print(f'Logging to {log_file}')
if opt.model_record:
    model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                 f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                 f'_layer_{opt.layer_num}_l2_{opt.l2}'

# loading data
data = pd.read_csv('./Data/' + opt.data + '.csv')
user = data['user_id'].unique()
item = data['item_id'].unique()
user_num = len(user)
item_num = len(item)
train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
train_set = myFloder(train_root, load_graphs)
test_set = myFloder(test_root, load_graphs)
if opt.val:
    val_set = myFloder(val_root, load_graphs)
# for data in train_set:
#     print(data[0][0])

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)
f = open(opt.data + '_neg', 'rb')
data_neg = pickle.load(f)  # 用于评估测试集
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True,
                        num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg),
                       pin_memory=True)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg),
                          pin_memory=True)

seq_dict, shortseq = data_process("./Data", opt.data)

train_set = []
for u in range(user_num):
    train_set.append(seq_dict[u][:-2])

# 滑动窗口生成项目初始图，adj表示
adj = slid_window(train_set, user_num, item_num, sw_width=5, n_out=5, in_start=0)  # 滑动窗口的数据是训练数据？是
adj = adj.to_sparse()

# 初始化模型
model = GOSR(device=device, seq_dict=seq_dict, shortseq=shortseq, adj=adj, user_num=user_num, item_num=item_num, input_dim=opt.hidden_size,
             item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop,
             user_long=opt.user_long, user_short=opt.user_short,
             item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update,
             item_update=opt.item_update, last_item=opt.last_item,
             layer_num=opt.layer_num).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()
best_result = [0, 0, 0, 0, 0, 0]  # recall,ndcg,hit5,hit10,hit20,mrr5,mrr10,mrr20
best_epoch = [0, 0, 0, 0, 0, 0]
best_loss = []
stop_num = 0
for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()
    # print(len(train_data))
    for user_index, user, batch_graph, label, last_item in train_data:
        # print("-----------------")
        # print(user_index)
        # print(user_index.shape)
        # print(last_item)
        # print("-----------------")
        iter += 1
        score = model(user_index.to(device), label.to(device), batch_graph.to(device), user.to(device),
                      last_item.to(device), epoch, iter, is_training=True)
        # print(score)
        loss = loss_func(score, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if iter % 400 == 0:
            print('Iter {}, loss {:.4f}'.format(iter, epoch_loss / iter), datetime.datetime.now())
            # break
    epoch_loss /= iter
    model.eval()
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

    # val
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_loss_all, top_val = [], []
        with torch.no_grad:
            for user_index, user, batch_graph, label, last_item, neg_tar in val_data:
                score, top = model(user_index.to(device), label.to(device), batch_graph.to(device), user.to(device),
                                   last_item.to(device), epoch, iter,
                                   neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                val_loss = loss_func(score, label.cuda())
                val_loss_all.append(val_loss.append(val_loss.item()))
                top_val.append(top.detach().cpu().numpy())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
            print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                  '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                  (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

    # test
    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_length = [], [], []
    iter = 0
    all_loss = []
    with torch.no_grad():
        for user_index, user, batch_graph, label, last_item, neg_tar in test_data:
            iter += 1
            score, top = model(user_index.to(device), label.to(device), batch_graph.to(device), user.to(device),
                               last_item.to(device), epoch, iter,
                               neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
            test_loss = loss_func(score, label.cuda())
            all_loss.append(test_loss.item())
            all_top.append(top.detach().cpu().numpy())
            all_label.append(label.numpy())
            if iter % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
        recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
        # rankedList5, rankedList10, rankedList20 = [], [], []
        # for index in range(len(all_top)):
        #     prediction = (-all_top[index]).argsort(1).argsort(1)
        #     predictions = prediction[:, 0]
        #     for i, rank in enumerate(predictions):
        #         if rank < 5:
        #             rankedList5.append(predictions[i])  # all_top要先排序再截取
        #         if rank < 10:
        #             rankedList10.append(predictions[i])
        #         if rank < 20:
        #             rankedList20.append(predictions[i])
        # print(rankedList10)
        # hr5, mrr5, ndcg5 = indicators_5(rankedList5, all_label, 5)
        # hr10, mrr10, ndcg10 = indicators_5(rankedList10, all_label, 10)
        # hr20, mrr20, ndcg20 = indicators_5(rankedList20, all_label, 20)
        best_loss.append([epoch, test_loss])  # 记录每个epoch损失
        if recall5 > best_result[0]:
            best_result[0] = recall5
            best_epoch[0] = epoch
            stop = False
        if recall10 > best_result[1]:
            if opt.model_record:
                torch.save(model.state_dict(), 'save_models/' + model_file + '.pkl')
            best_result[1] = recall10
            best_epoch[1] = epoch
            stop = False
        if recall20 > best_result[2]:
            best_result[2] = recall20
            best_epoch[2] = epoch
            stop = False
            # ------select Mrr------------------
        if ndgg5 > best_result[3]:
            best_result[3] = ndgg5
            best_epoch[3] = epoch
            stop = False
        if ndgg10 > best_result[4]:
            best_result[4] = ndgg10
            best_epoch[4] = epoch
            stop = False
        if ndgg20 > best_result[5]:
            best_result[5] = ndgg20
            best_epoch[5] = epoch
            stop = False
        # if hr5 > best_result[6]:
        #     best_result[6] = hr5
        #     best_epoch[6] = epoch
        #     stop = False
        # if hr10 > best_result[7]:
        #     best_result[7] = hr10
        #     best_epoch[7] = epoch
        #     stop = False
        # if hr20 > best_result[8]:
        #     best_result[8] = hr20
        #     best_epoch[8] = epoch
        #     stop = False
        # if mrr5 > best_result[9]:
        #     best_result[9] = mrr5
        #     best_epoch[9] = epoch
        #     stop = False
        # if mrr10 > best_result[10]:
        #     best_result[10] = mrr10
        #     best_epoch[10] = epoch
        #     stop = False
        # if mrr20 > best_result[11]:
        #     best_result[11] = mrr20
        #     best_epoch[11] = epoch
        #     stop = False
        if stop:
            stop_num += 1
        else:
            stop_num = 0
        print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f\t'
              '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
              (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
               best_result[4], best_result[5], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
        print(best_loss)

    # plt.plot(epoch_loss, label='loss')
    # plt.plot(all_loss, label='val_loss')
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('./loss.png')

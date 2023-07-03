import os
import math
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
import dgl
import torch
import numpy as np
import pandas as pd


def pickle_loader(path):
    a = cPickle.load(open(path, 'rb'))
    return a


def user_neg(data, item_num):
    item = range(item_num)
    def select(data_u, item):  # 找到2个数组中集合元素的差异。返回值：在ar1中但不在ar2中的已排序的唯一值。
        return np.setdiff1d(item, data_u)
    return data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))


def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


class myFloder(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        # print(data)
        return data

    def __len__(self):
        return self.size


def collate(data):
    user = []
    graph = []
    last_item = []
    label = []
    for da in data:
        user.append(da[0])
        graph.append(da[1])
        last_item.append(da[2])
        label.append(da[3])
    return torch.Tensor(user).long(), dgl.batch_hetero(graph), torch.Tensor(last_item).long(), torch.Tensor(label).long()


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir


def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user_alis = []
    graph = []
    last_item = []
    label = []
    user = []
    length = []
    for da in data:
        user_alis.append(da[0])
        graph.append(da[1])
        last_item.append(da[2])
        label.append(da[3])
        user.append(da[4])
        length.append(da[5])
    return torch.Tensor(user_alis).long(), dgl.batch_hetero(graph), torch.Tensor(last_item).long(), \
           torch.Tensor(label).long(), torch.Tensor(length).long(), torch.Tensor(neg_generate(user, user_neg)).long()


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def eval_metric(all_top, all_label, all_length, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        per_length = all_length[index]
        if random_rank:
            prediction = (-all_top[index]).argsort(1).argsort(1)
            predictions = prediction[:, 0]
            for i, rank in enumerate(predictions):
                # data_l[per_length[i], 6] += 1
                if rank < 20:
                    ndgg20.append(1 / np.log2(rank + 2))
                    recall20.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 5] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 2] += 1
                    # else:
                    #     data_l[99, 5] += 1 / np.log2(rank + 2)
                    #     data_l[99, 2] += 1
                else:
                    ndgg20.append(0)
                    recall20.append(0)
                if rank < 10:
                    ndgg10.append(1 / np.log2(rank + 2))
                    recall10.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 4] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 1] += 1
                    # else:
                    #     data_l[99, 4] += 1 / np.log2(rank + 2)
                    #     data_l[99, 1] += 1
                else:
                    ndgg10.append(0)
                    recall10.append(0)
                if rank < 5:
                    ndgg5.append(1 / np.log2(rank + 2))
                    recall5.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 3] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 0] += 1
                    # else:
                    #     data_l[99, 3] += 1 / np.log2(rank + 2)
                    #     data_l[99, 0] += 1
                else:
                    ndgg5.append(0)
                    recall5.append(0)

        else:
            for top_, target in zip(all_top[index], all_label[index]):
                recall20.append(np.isin(target, top_))
                recall10.append(np.isin(target, top_[0:10]))
                recall5.append(np.isin(target, top_[0:5]))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg20.append(0)
                else:
                    ndgg20.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg10.append(0)
                else:
                    ndgg10.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg5.append(0)
                else:
                    ndgg5.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
    #pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n10','number']).to_csv(name+'.csv')
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20), \
           pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n20','number'])



def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def data_process(path, dataset):
    print("Loading {} dataset...".format(dataset))

    # 将时间戳转换为时间
    # with open("{}/{}.csv".format(path, dataset), 'rb') as f:
    #     f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
    #     for line in f:
    #         line = line.decode().rstrip().split('\t')
    #         timestamp =line[2]
    #         # 转换成localtime
    #         timestamp=float(timestamp)
    #         time_local = time.localtime(timestamp)
    #         # 转换成新的时间格式(2016-05-05 20:28:54)
    #         dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    #         line[2]=dt
    if dataset in ['Games']:
        filename = './Data/Games.csv'
        data = pd.read_csv('./Data/Games.csv')
        user = data['user_id'].unique()
        item = data['item_id'].unique()
        total = sum(1 for line in open(filename))
        print('The total lines is ', total)  # total是文件总交互数
        user_num = len(user)  # m
        item_num = len(item)  # n

        result = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                result[int(line[0])] = []

        seq_dict = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # result中的元素也是一个字典的形式
                seq_dict[int(line[0])] = []  # 定义嵌套字典
        # 定义一个记录用户第几个键值对的数组
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # 嵌套字典的赋值方式
                # seq_dict[int(line[0])][int(line[1])] = int(line[2])  # 计算曼哈顿距离
                seq_dict[int(line[0])].append((int(line[1]), int(line[2])))
        # print(seq_dict)
        # print(seq_dict[0][0][0])

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)
        shortseq=seq_dict  # 记录未固定长度的序列情况

        # 统计
        count = 0
        for i in range(len(seq_dict)):
            if len(seq_dict[i]) > 15:
                count += 1
        print(count)

        # 固定序列长度
        for user in range(len(seq_dict)):
            if len(seq_dict[user]) < 15:  # 填充
                gap = 15-len(seq_dict[user])
                for i in range(gap):  # 直接填充
                    seq_dict[user].append((seq_dict[user][0][0], seq_dict[user][0][1]))
            elif len(seq_dict[user]) == 15:
                seq_dict[user] = seq_dict[user]
            else:
                seq_dict[user] = seq_dict[user][-15:]  # 保留后10个

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)

    if dataset in ['Movie']:
        filename = './Data/Movie1.csv'
        data = pd.read_csv('./Data/Movie1.csv')
        user = data['user_id'].unique()
        item = data['item_id'].unique()
        total = sum(1 for line in open(filename))
        print('The total lines is ', total)  # total是文件总交互数
        user_num = len(user)  # m
        item_num = len(item)  # n

        result = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                result[int(line[0])] = []

        seq_dict = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # result中的元素也是一个字典的形式
                seq_dict[int(line[0])] = []  # 定义嵌套字典
        # 定义一个记录用户第几个键值对的数组
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # 嵌套字典的赋值方式
                # seq_dict[int(line[0])][int(line[1])] = int(line[2])  # 计算曼哈顿距离
                seq_dict[int(line[0])].append((int(line[1]), int(line[2])))
        # print(seq_dict)
        # print(seq_dict[0][0][0])

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)
        shortseq=seq_dict  # 记录未固定长度的序列情况

        # 统计
        count = 0
        for i in range(len(seq_dict)):
            if len(seq_dict[i]) > 50:
                count += 1
        print(count)

        # 固定序列长度
        for user in range(len(seq_dict)):
            if len(seq_dict[user]) < 50:  # 填充
                gap = 50-len(seq_dict[user])
                for i in range(gap):  # 直接填充
                    seq_dict[user].append((seq_dict[user][0][0],seq_dict[user][0][1]))
            elif len(seq_dict[user]) == 50:
                seq_dict[user] = seq_dict[user]
            else:
                seq_dict[user] = seq_dict[user][-50:]  # 保留后8个

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)

    # TODO 这里是现在这个数据集的情况，改的地方在下面TODO list
    if dataset in ['Lastfm1']:
        filename = './Data/Lastfm1.csv'
        data = pd.read_csv('./Data/Lastfm1.csv')
        user = data['user_id'].unique()
        item = data['item_id'].unique()
        total = sum(1 for line in open(filename))
        print('The total lines is ', total)  # total是文件总交互数
        user_num = len(user)  # m
        item_num = len(item)  # n

        result = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                result[int(line[0])] = []

        seq_dict = {}
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # result中的元素也是一个字典的形式
                seq_dict[int(line[0])] = []  # 定义嵌套字典
        # 定义一个记录用户第几个键值对的数组
        with open("{}/{}.csv".format(path, dataset), 'rb') as f:
            f.readline()  # 用于从文件读取整行，包括 '\n' 字符。
            for line in f:
                line = line.decode().rstrip().split(',')
                # 嵌套字典的赋值方式
                # seq_dict[int(line[0])][int(line[1])] = int(line[2])  # 计算曼哈顿距离
                seq_dict[int(line[0])].append((int(line[1]), int(eval(line[2]))))
        # print(seq_dict)
        # print(seq_dict[0][0][0])

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)
        shortseq=seq_dict  # 记录未固定长度的序列情况

        # 统计
        count = 0
        for i in range(len(seq_dict)):
            if len(seq_dict[i]) > 120:
                count += 1
        print(count)

        # TODO 每个用户取多少固定序列，这里没更新，现在取120
        # 固定序列长度
        for user in range(len(seq_dict)):
            if len(seq_dict[user]) < 120:  # 填充
                gap = 120 - len(seq_dict[user])
                for i in range(gap):  # 直接填充
                    seq_dict[user].append((seq_dict[user][0][0], seq_dict[user][0][1]))
            elif len(seq_dict[user]) == 120:
                seq_dict[user] = seq_dict[user]
            else:
                seq_dict[user] = seq_dict[user][-120:]  # 保留后8个

        # 按照时间顺序排列交互
        for user in range(user_num):
            seq_dict[user] = sorted(seq_dict[user], key=lambda x: x[1], reverse=False)
        # print(seq_dict)

    return seq_dict, shortseq


def slid_window(train_data, user_num, item_num, sw_width=5, n_out=5, in_start=0):
    # 该函数实现窗口大小为5、滑动步长为1的滑动窗口截取序列数据
    data = train_data
    adj = torch.zeros((item_num, item_num))

    for u in range(user_num):
        # TODO 窗口应该滑动多少次，应该是序列长度120-大小
        for in_start in range(120-sw_width):
            in_end = in_start + sw_width  # 14
            out_end = in_end + n_out  # 19
            # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
            if out_end < len(data[u])+n_out+2:  # 18+2=20
                # 训练数据以滑动步长1截取
                train_seq = data[u][in_start:in_end]  # 9-13,14
                # train_seq = train_seq.reshape((len(train_seq), 1))
                for it1 in train_seq:
                    for it2 in train_seq:
                        # TODO 设定的时间阈值，也就是如果两个项目发生的时间小于阈值就有边
                        if it2[1] - it1[1] <= 10000000:  # 阈值待定
                            adj[int(it1[0])][int(it2[0])] = 1  # int(it1)代表项目
                            adj[int(it2[0])][int(it1[0])] = 1  # 关系对称
            else:
                continue
        # in_start += 1
    print(adj.to_sparse())
    return adj


def position_encode(object_num, d_model, max_len=88888):
    max_len = object_num
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe[:object_num, 0, :]


def HR(truth, pred, N):
    """
    Hit Ratio
    truth: the index of real target (from 0 to len(truth)-1)
    pred: rank list
    N: top-N elements of rank list
    """
    hit_N = 0.0
    for u in range(len(pred)):
        top_N = pred[u].squeeze()[:N]
        truth_u = truth[u]
        if truth_u in top_N:
            hit_N = hit_N + 1.0
        else:
            hit_N = hit_N + 0.0

    return hit_N/len(pred)


def indicators_5(rankedList, testList):
    Hits_i = 0
    Len_R = 0
    Len_T = len(testList)
    MRR_i = 0
    HR_i = 0
    NDCG_i = 0
    for i in range(len(rankedList)):
        for j in range(len(rankedList[i])):
            if testList[i][0] == rankedList[i][j]:
                Hits_i += 1
                HR_i += 1
                # 注意j的取值从0开始
                MRR_i += 1/(j+1)
                NDCG_i += 1/(math.log2(1+j+1))
                break
    HR_i /= Len_T
    MRR_i /= Len_T
    NDCG_i /= Len_T
    print(Hits_i)
    # print(f'HR@5={HR_i}')
    # print(f'MRR@5={MRR_i}')
    # print(f'NDCG@5={NDCG_i}')
    return HR_i, MRR_i, NDCG_i

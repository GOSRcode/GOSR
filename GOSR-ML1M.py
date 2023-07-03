import gc
import dgl
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from utils import slid_window, position_encode

class NewRnn(nn.Module):
    def __init__(self, device, feature_dim, hidden_size, sequence_length=10, dropout=0.1, batch_size=1):
        # rnn部分
        super().__init__()
        self.input_size = feature_dim
        self.hidden_size = hidden_size
        self.nonlinearity = 'tanh'
        self.bias = True
        self.batch_first = True
        self.dropout = dropout
        self.bidirectional = False
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.rnn = nn.RNN(self.input_size, self.hidden_size)

    def forward(self, feature, item_embedding):  # feature代表一个序列，seq_dict，每一个序列执行一次。
        hidden_prev = torch.randn(self.batch_size, 1, self.hidden_size)
        hidden_prev = hidden_prev.to(self.device)
        for i in range(len(feature)):
            index = feature[i][0]
            item = item_embedding[index].unsqueeze(0).unsqueeze(0)  # input代表该项目的嵌入。所以在RNN中数据通常是三维的
            item = torch.tensor(item, dtype=torch.float32)
            output, hidden_prev = self.rnn(item, hidden_prev)
            hidden_prev = hidden_prev * np.reciprocal(feature[i][1] - feature[i - 1][1]) + hidden_prev * np.reciprocal(
                1)
            item_embedding[index] = output.squeeze(0).squeeze(0)
        return item_embedding


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, device):
        support = torch.mm(x, self.weight)
        adj = adj.to(device)
        support = support.to(device)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, feature_dim, output_dim, hidden_size, device, dropout=0.1, activation="relu"):
        """
        :param feature_dim:
        :param output_dim:
        :param hidden_size:
        :param dropout:
        :param activation:
        """
        super().__init__()
        self.device = device
        self.conv1 = GraphConvolution(feature_dim, hidden_size, device)
        self.conv2 = GraphConvolution(hidden_size, output_dim, device)
        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):  # feature用item_embedding
        """
        :param feature: shape(num*dim)
        :param adj: shape(num*num)
        :return:
        """
        x1 = self.activation(self.conv1(feature, adj, self.device))
        x1 = F.dropout(x1, p=self.dropout)
        x2 = self.conv2(x1, adj, self.device)
        x2 = F.log_softmax(x2, dim=1)
        return x2


class GOSR(nn.Module):
    def __init__(self, device, seq_dict, shortseq, adj, user_num, item_num, input_dim, item_max_length, user_max_length,
                 feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True):
        super(GOSR, self).__init__()
        self.device = device
        self.seq_dict = seq_dict
        self.shortseq = shortseq
        self.adj = adj
        self.adj_i = adj
        # self.adj_u=torch.zeros((50,50))
        # self.adj_u = torch.zeros((user_num, user_num))
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        # long- and short-term encoder
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # update function
        self.user_update = user_update
        self.item_update = item_update

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.hidden_size, self.user_max_length,
                                                self.item_max_length, feat_drop, attn_drop,
                                                self.user_long, self.user_short, self.item_long, self.item_short,
                                                self.user_update, self.item_update) for _ in range(self.layer_num)])
        # self.rnn = NewRnn(device=device, feature_dim=self.hidden_size, hidden_size=self.hidden_size)
        self.gcn = GCN(feature_dim=self.hidden_size, hidden_size=self.hidden_size * 2, output_dim=self.hidden_size,
                       device=device)
        self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.a = 0.35
        # self.a = nn.parameter.Parameter(torch.rand(1), requires_grad=True)
        self.y = 0.6

        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2)
        self.bert = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1)
        self.flag = True

        self.reset_parameters()

        # self.relation_update1 = nn.Linear(self.item_num, self.item_num, bias=False)
        # self.relation_update2 = nn.Linear(self.item_num, self.item_num, bias=False)
        # self.relation_update3 = nn.Linear(self.item_num, self.item_num, bias=False)
        # self.relation_update4 = nn.Linear(self.item_num, self.item_num, bias=False)

    def slid_window(self, train_data, user_num, item_num, sw_width=5, n_out=5, in_start=0):
        # 该函数实现窗口大小为5、滑动步长为1的滑动窗口截取序列数据
        # n_out不知道之前代码代表什么含义,in_start之前代码也没使用
        data = train_data
        adj = torch.zeros((item_num, item_num))
        for u in range(user_num):  # 遍历每一个user
            # 这样遍历能更方便的获得滑动窗口内项目之间的距离
            for sw in range(2, sw_width + 1):
                for start in range(0, len(data[u]) - sw + 1 - 2):
                    vi = data[u][start]
                    vj = data[u][start + sw - 1]
                    adj[vi[0]][vj[0]] += self.fun(vj[1] - vi[1])
                    adj[vj[0]][vi[0]] += self.fun(vj[1] - vi[1])
        print(adj.to_sparse())
        return adj

        # fun是一个关于两个项目之间距离和时间间隔的函数(权值)
        # 距离越远，fun越小，时间间隔越大，fun越小

    def fun(self, t):
        # 可以用其他函数，比如log的倒数
        # 取a,b两个数来平衡距离和时间间隔对adj的影响权重
        a = 1
        b = 0.1
        return a / (t + 1)  # 最简单的直接取倒数

        # 归一化：就是把矩阵的值按比例压缩到[0,1]范围内，也就是(x-min)/(max-min)

    def normalization(self, data):
        _range = data.max() - data.min()
        return (data - data.min()) / _range

    def position_encode(self, object_num, d_model, max_len=88888):
        max_len = object_num
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe[:object_num, 0, :]

    def position_encode2(self, n, t, d):
        """
        n:项目的下标，也就是ri中的i，也是公式里面的2j或者2j+1
        t:两个项目的时间差，也就是公式里面的tn-ti
        d:嵌入向量的维度，公式中的d
        """
        x = math.pow(10000, n / d)  # 计算10000的n/d次方
        if n % 2 == 0:
            ans = math.sin(t / x)
        else:
            ans = math.cos(t / x)
        return ans

    def calculate_cosine_similarity_matrix(self, h_emb, eps=1e-8):
        h_emb = h_emb.cpu()

        elements = torch.tensor([])
        elements = elements.to(self.device)
        r = 10
        eve = self.item_num // r  # 每一部分的项目数
        for i in range(r + 1):
            # MPAN相对间隔
            if i != r:
                ran = range(eve * i, eve * (i + 1))
                ss=eve * i
            else:
                ran = range(self.item_num - eve, self.item_num)
                ss=self.item_num - eve
            print(ran)

            # relative_time = torch.zeros(self.item_num // r, self.item_num)
            # for seq in range(len(self.seq_dict)):  # 对每个序列计算相对时间间隔
            #     t = self.seq_dict[seq][len(self.seq_dict[seq]) - 3]
            #     for z in range(len(self.seq_dict[seq]) - 2):
            #         # 对position_encode存疑
            #         # relative_time[seq_dict[seq][i][0]][t[0]] += position_encode(t-seq_dict[seq][i][1])
            #         if self.seq_dict[seq][z][0] in ran:
            #             relative_time[self.seq_dict[seq][z][0] - ss][t[0]] += self.position_encode2(z,
            #                                                                                              t[1] -
            #                                                                                              self.seq_dict[
            #                                                                                                  seq][z][1],
            #                                                                                              self.hidden_size)
            # relative_time = relative_time.to_sparse()
            # relative_time = relative_time.to(self.device)
            # print("relative_time ")
            # print(relative_time)
            #
            # gongxian = torch.zeros(self.item_num // r, self.item_num)
            # # HG-GNN的共现   这里seq_dict应该是未填充的不固定长度的
            # for seq in range(len(self.shortseq)):
            #     items = set()
            #     for c in range(len(self.shortseq[seq]) - 2):
            #         items.add(self.shortseq[seq][c][0])
            #     items = list(items)
            #     for z in range(len(items)):
            #         for j in range(len(items)):
            #             if z != j and items[z] in ran:  # 不会认为自己和自己是共现的
            #                 gongxian[items[z] - ss][items[j]] += 1 / len(self.shortseq[seq])
            # gongxian = gongxian.to_sparse()
            # gongxian = gongxian.to(self.device)
            # print("gongxian")
            # print(gongxian)
            #
            # # 共现矩阵(无滑动窗口版)
            # C = torch.zeros(self.item_num // r, self.item_num)
            # # C = C.cpu()
            # for seq in range(self.user_num):
            #     for z in range(0, len(self.seq_dict[seq]) - 2):
            #         for j in range(z, len(self.seq_dict[seq]) - 2):
            #             if z != j and self.seq_dict[seq][z][0] in ran:
            #                 C[self.seq_dict[seq][z][0] - ss][self.seq_dict[seq][j][0]] += 1
            #                 # C[self.seq_dict[seq][j][0]][self.seq_dict[seq][i][0]] += 1
            # C = C.to_sparse()
            # C = C.to(self.device)
            # print("C:")
            # print(C)
            #
            # # # 加滑动窗口共现
            # # C = torch.zeros(self.item_num // r, self.item_num)
            # # # C = C.cpu()
            # # sw_width = 5
            # # for u in range(self.user_num):  # 遍历每一个user
            # #     for sw in range(2, sw_width + 1):
            # #         for start in range(0, len(self.seq_dict[u]) - sw + 1 - 2):
            # #             vi = self.seq_dict[u][start]
            # #             vj = self.seq_dict[u][start + sw - 1]
            # #             if vi[0] != vj[0] and vi[0] in ran:
            # #                 C[vi[0]-ss][vj[0]] += self.fun(vj[1] - vi[1])
            # # C = C.to_sparse()
            # # C = C.to(self.device)
            # # print("C:")
            # # print(C)
            #
            # # 相对位置 这里变成不对称关系
            # jiange = torch.zeros(self.item_num // r, self.item_num)
            # # jiange = jiange.cpu()
            # for seq in range(self.user_num):
            #     for z in range(0, len(self.seq_dict[seq]) - 2):
            #         for j in range(z, len(self.seq_dict[seq]) - 2):
            #             if self.seq_dict[seq][z][0] in ran:
            #                 jiange[self.seq_dict[seq][z][0] - ss][self.seq_dict[seq][j][0]] += j - z
            #                 # jiange[self.seq_dict[seq][j][0]][self.seq_dict[seq][i][0]] += j - i
            #
            # jiange = jiange.to(self.device)
            # jiange = torch.div(jiange, C.to_dense())
            # for z in range(jiange.shape[0]):
            #     for j in range(jiange.shape[1]):
            #         if np.isnan(jiange[z][j].cpu()):
            #             jiange[z][j] = 0
            # # 映射函数加在这里
            # for z in range(jiange.shape[0]):
            #     for j in range(jiange.shape[1]):
            #         if jiange[z][j] != 0:
            #             jiange[z][j] = math.exp(-jiange[z][j])
            # # TODO jiange先不要softmax了
            # # jiange = F.softmax(jiange)
            # jiange = jiange.to_sparse()
            # jiange = jiange.to(self.device)
            # # jiange = jiange.cpu()
            # print("jiange:")
            # print(jiange)
            #
            # # C = C.to_dense()
            # # TODO C不要softmax了
            # # C = F.softmax(C)
            # # C = C.to_sparse()
            # C = C.to(self.device)
            # print("C:")
            # print(C)

            # 每个因素设置相同的权值（不同的也行，但调参麻烦）
            # a1 = a2 = a3 = a4 = 1 / 4
            # elements = a1 * self.normalization(relative_time) + a2 * self.normalization(gongxian) \
            #            + a3 * self.normalization(C) + a4 * self.normalization(jiange)

            relative_time = torch.load("./Data/relamov.pt")[ran]
            relative_time = relative_time.to_sparse()
            relative_time = relative_time.to(self.device)

            gongxian = torch.load("./Data/goximov.pt")[ran]
            gongxian = gongxian.to_sparse()
            gongxian = gongxian.to(self.device)

            # C = torch.load("./Data/Cmov.pt")[ran]
            # C = C.to_sparse()
            # C = C.to(self.device)

            # 加滑动窗口共现
            C = torch.load("./Data/C2mov.pt")[ran]
            C = C.to_sparse()
            C = C.to(self.device)

            jiange = torch.load("./Data/jigemov.pt")[ran]
            jiange = jiange.to_sparse()
            jiange = jiange.to(self.device)

            # element = self.relation_update1(relative_time.to_dense()) + self.relation_update2(
                # gongxian.to_dense()) + self.relation_update3(C.to_dense()) + self.relation_update4(jiange.to_dense())
            element = F.softmax((relative_time+gongxian+C+jiange).to_dense())
            if i != r:
                elements = torch.cat((elements, element), 0)
            else:
                elements = torch.cat((elements, element[-(self.item_num - eve * r):]), 0)
        print(elements.shape)
        relative_time = []
        gongxian = []
        C = []
        # C2=[]
        jiange = []
        print("elements:")
        print(elements)
        elements = self.normalization(elements)
        # elements = elements.to_sparse()
        elements = elements.cpu()
        print("elements:")
        print(elements)

        # 按原来方法计算sim_matrix
        a_n = h_emb.norm(dim=1).unsqueeze(1)
        a_n = a_n.cpu()
        a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))
        a_norm = a_norm.cpu()

        # cosine similarity matrix
        sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0, 1))
        sim_matrix = sim_matrix.cpu()

        # 然后用相加的方式更新sim_matrix
        sim_matrix = self.normalization(sim_matrix + elements)
        elements = []

        print("sim_matrix")
        print(sim_matrix)
        # sim_matrix=sim_matrix+C+jiange+seppos

        return sim_matrix


    def sim_item(self, item_embedding, adj, epoch):  # TODO 最好在每个方法进行说明，相似度计算。再详细一点可以对输入的参数进行说明。
        # TODO 相似度的计算方法
        adj = adj.cpu()
        # adj = adj.to(self.device)
        adj = adj.to_dense()
        adj_temp = self.calculate_cosine_similarity_matrix(item_embedding)

        adj = self.a * adj + (1 - self.a) * adj_temp
        b = 500000 - int(epoch * 100000)
        b = torch.topk(adj.reshape(-1), b)[0][-1]

        adj[adj < b] = 0

        adj = adj.to_sparse()

        return adj.to(self.device)

    def sim_user(self, user_embedding, epoch):

        adj = self.calculate_cosine_similarity_matrix(user_embedding)

        c = self.y + epoch / 3 * 0.01  # 越来越大，跨度不能太快
        adj[adj < c] = 0
        adj = adj.to_sparse()

        return adj.to(self.device)

    def user_initial(self, user_embedding):
        adj = self.calculate_cosine_similarity_matrix(user_embedding)
        adj[adj < self.y] = 0
        adj = adj.to_sparse()

        return adj

    def forward(self, user, label, g, user_index=None, last_item_index=None, epoch=0, iter=0, neg_tar=None, is_training=False):
        # print(iter)
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())

        layers = 2
        for l in range(layers):
            """"加在这里"""
            tmp_embedding = nn.Embedding.from_pretrained(self.item_embedding.weight)  # nn.Embedding.weight随机初始化方式是标准正态分布,残差

            it_it_embedding = tmp_embedding.weight.clone()
            hn = torch.randn(2, 1, self.hidden_size)
            hn = hn.to(self.device)
            it_list = set()
            if torch.is_tensor(user):
                user = user.cpu().numpy()
            for c, u in enumerate(user):
                seq_dict = []
                u = user[c]
                for i in range(len(self.seq_dict[u])):
                    if self.seq_dict[u][i][0] != label[c]:
                        seq_dict.append(self.seq_dict[u][i])
                        it_list.add(self.seq_dict[u][i][0])
                    else:
                        break
                # print(u)
                # print(label[c])
                # print(seq_dict)
                # print(self.seq_dict[u])
                # 1/0
                for j in range(len(seq_dict)):
                    index = seq_dict[j][0]
                    item = tmp_embedding.weight[index].unsqueeze(0).unsqueeze(0)
                    item = torch.tensor(item, dtype=torch.float32)
                    item, hn = self.rnn(item, hn)
                    item = F.softmax(item)
                    # hn = hn * np.reciprocal(seq_dict[j][1] - seq_dict[j - 1][1]) + hn * np.reciprocal(1)
                    hn = F.softmax(hn)
                    # print("hn", hn)
                    it_it_embedding[index] = item.squeeze(0).squeeze(0)

            it_it_embedding = it_it_embedding + tmp_embedding.weight
            it_list = list(sorted(it_list))
            it_emb = it_it_embedding[it_list]
            adj_i = self.adj_i.to_dense()
            adj_i = adj_i[it_list]
            adj_i = adj_i[:, it_list]

            it_emb = self.gcn(it_emb, adj_i)
            it_emb = F.softmax(it_emb, dim=-1)
            # 更新到总的item_embedding里面
            for i, val in enumerate(it_list):
                it_it_embedding[val] = it_emb[i]
            # F.softmax(self.item_update(torch.cat([tmp_embedding.weight, it_it_embedding], -1)), -1)
            self.item_embedding = nn.Embedding.from_pretrained(F.softmax(self.item_update(torch.cat([tmp_embedding.weight, it_it_embedding], -1)), -1))

            # 最后一个iter验证加强关系 TODO
            if iter == 3830:
                self.adj_i = self.sim_item(torch.tensor(self.item_embedding.weight), self.adj_i, epoch)
                print(self.adj_i)

        if self.layer_num > 0:  # 3
            for conv in self.layers:  # self.layers包括三层GNN层
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))

            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)

        unified_embedding = self.unified_map(torch.cat(user_layer, -1))  # 在给定维度上对输入的张量序列seq 进行连接操作。

        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))

        # print("item_embedding", self.item_embedding.weight)
        # print("unified_embedding", unified_embedding)
        # print("score", score)

        # print(self.item_embedding.weight.shape)
        # print(unified_embedding.shape)
        # print(score.shape)

        adj_i = None
        del adj_i
        gc.collect()
        torch.cuda.empty_cache()

        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class DGSRLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att',
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', K=4):
        super(DGSRLayers, self).__init__()
        self.hidden_size = in_feats
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K).cuda()
        if self.user_long in ['orgat', 'gcn', 'gru'] and self.user_short in ['last', 'att', 'att1']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        # attention+ attention mechanism
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.item_long in ['orgat']:
            self.i_time_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
            self.i_time_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
            self.u_time_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)

        self.flag = True

    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):
        # user_encoder 对user进行编码
        # update all nodes
        # DGL中针对异质图进行消息传递的函数。
        # muti_update_all()中先是逐关系做了update_all()操作，然后再跨类型操作。update_all()更新的是所有终点类型的节点，而不是所有节点
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        # 先根据time排序
        # order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['item_h'].shape[0]
        # 长期兴趣编码,可以先gru，再att
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'],
                             dim=2) \
                   / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        # 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
                torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_i(torch.cat(h, -1))}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        # 先根据time排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        # if self.flag:
        #     print('nodes:{}'.format(nodes))
        #     print('time:{}'.format(nodes.mailbox['time']))
        #     print('item_h:'.format(nodes.mailbox['item_h']))
        #     print('user_h:'.format(nodes.mailbox['user_h']))
        #     self.flag = False
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['user_h'].shape[0]
        # 长期兴趣编码
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) * nodes.mailbox['user_h'],
                             dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        # 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2) / torch.sqrt(
                torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h, -1))}


def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    # tmp = np.roll(np.cumsum(b_user_size).cpu(), 1)
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_user_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_user_index = torch.Tensor(tmp).long().cuda() + user_index
    # ----pytorch写法
    # 顺移。input是咱们要移动的tensor向量，shifts是要移动到的位置，要移动去哪儿，dims是值在什么方向上(维度)去移动。
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)  # 返回维度dim中输入元素的累计和。【功能：累加】
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]


def graph_item(bg, last_index, item_embedding):  # 图，项目索引，项目嵌入
    b_item_size = bg.batch_num_nodes('item')
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_item_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_item_index = torch.Tensor(tmp).long().cuda() + last_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]


def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic


def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user).long(), torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long()


def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[int(u)], neg_num, replace=False)
    return neg


def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user_index = []
    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user_index.append(da[1]['user'])
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user_index).long(), torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(
        last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long()

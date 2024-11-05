import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn.init import kaiming_uniform_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LTLayer(nn.Module):
    def __init__(self, in_channels, dc):
        super(LTLayer, self).__init__()

        self.in_channels = in_channels
        self.dc = dc
        self.w = nn.Parameter(torch.rand((self.in_channels, self.in_channels)))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def forward(self, x, mask=None):
        # x = x.detach()
        s = torch.exp(x @ self.w @ torch.transpose(x, dim0=-2, dim1=-1) / self.dc)
        s = self.relu(s)

        if mask is not None:
            s = s * mask.view(x.shape[0], x.shape[1]).to(s.dtype)

        clean_diag = torch.diag(torch.ones(x.shape[-2])).to(torch.bool).to(device)
        s = s * (~clean_diag)

        return s

    def reset_parameters(self) -> None:
        stdv = 1. / sqrt(self.in_channels)
        self.w.data.uniform_(-stdv, stdv)
        self.w.data += torch.eye(self.in_channels)


class LTCluster:
    def __init__(self, s, edge_index, lt_num):
        self.lt_num = lt_num
        self.s = s.detach().cpu()
        self.edge_index = edge_index.detach().cpu()
        self.num = s.shape[0]

        self.density = None
        self.Q = None
        self.delta = torch.zeros(self.num, requires_grad=False)
        self.Pa = torch.zeros((2, self.num), dtype=torch.int64, requires_grad=False)
        self.gamma = None
        self.treeID = torch.zeros(self.num, dtype=torch.int, requires_grad=False) - 1
        self.c_id = None
        self.AL = [[] for i in range(self.lt_num)]

    def local_density(self):
        self.density = self.s.sum(axis=1).detach()
        self.Q = torch.argsort(self.density, descending=True)

    def leading_node(self):
        self.Pa[1, self.Q[0]] = -1
        self.Pa[0, self.Q[0]] = self.Q[0]
        self.delta[self.Q[0]] = self.density[self.Q[0]]

        for i in range(1, self.num):
            neighbor = self.edge_index[1][self.edge_index[0] == self.Q[i]] # Q[i]结点的邻接结点
            sim = self.s[self.Q[i]]  # Q[i]结点与其他结点的相似度
            # assert neighbor.size(0) > 0
            if neighbor.size(0) == 0:
                neighbor = self.Q[i - 1].reshape(1, )
                # self.Pa[1, self.Q[i]] = -2
                # self.delta[self.Q[i]] = 0
            self.Pa[1, self.Q[i]] = neighbor[torch.argmax(sim[neighbor])]  # Q[i]结点指向的引领结点
            self.Pa[0, self.Q[i]] = self.Q[i]  # 记录边的起始结点，方便后续子树划分
            self.delta[self.Q[i]] = self.s[self.Q[i], self.Pa[1, self.Q[i]]]

    def center(self):
        self.gamma = torch.argsort(self.density * self.delta, descending=True)
        self.c_id = self.gamma[:self.lt_num]

    def get_subtree(self):
        for i in range(self.lt_num):
            curInd = int(self.c_id[i])
            self.AL[i].append(curInd)
            self.treeID[curInd] = i
            q = (self.Pa[0][self.Pa[1] == curInd]).tolist()  # BFS
            while len(q) > 0:
                curInd = q.pop(0)
                if self.treeID[curInd] == -1:
                    self.AL[i].append(int(curInd))
                    self.treeID[curInd] = i
                    child = self.Pa[0][self.Pa[1] == curInd]
                    q += child.tolist()
        remain = torch.nonzero(self.treeID == -1).flatten()
        for i in range(len(remain)):
            lt = i % self.lt_num
            self.treeID[remain[i]] = lt
            self.AL[lt].append(int(remain[i]))
            self.Pa[1, remain[i]] = self.c_id[lt]
            self.delta[remain[i]] = self.s[remain[i], self.c_id[lt]]

    def GetSubtreeR(self):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(self.lt_num):
            self.AL[i].append(int(self.c_id[i]))

        for i in range(self.lt_num):
            self.treeID[self.c_id[i]]=i

        for nodei in range(self.num): ### casscade label assignment
            curInd = self.Q[nodei]
            if self.treeID[curInd] > -1:
                continue
            else:
                paID = self.Pa[1, curInd]
                curTreeID = self.treeID[paID]
                self.treeID[curInd] = curTreeID
                self.AL[curTreeID].append(int(curInd))

        remain = torch.nonzero(self.treeID == -1).flatten()
        for i in range(len(remain)):
            lt = i % self.lt_num
            self.treeID[remain[i]] = lt
            self.AL[lt].append(int(remain[i]))
            self.Pa[1, remain[i]] = self.c_id[lt]
            self.delta[remain[i]] = self.s[remain[i], self.c_id[lt]]

    def construct(self):
        t1 = time.time()
        self.local_density()
        t2 = time.time()
        # print('local_density cost: {:.4f}s'.format(t2-t1))
        self.leading_node()
        t3 = time.time()
        # print('leading_node cost: {:.4f}s'.format(t3 - t2))
        self.center()
        t4 = time.time()
        # print('center cost: {:.4f}s'.format(t4 - t3))
        self.get_subtree()
        # self.GetSubtreeR()
        t5 = time.time()
        # print('GetSubtreeR cost: {:.4f}s'.format(t5 - t4))

        return self.c_id, self.treeID, self.AL

    def find_root(self, cur_id):
        if self.treeID[cur_id] != -1:
            return self.treeID[cur_id]

        self.treeID[cur_id] = self.find_root(self.Pa[cur_id])
        return self.treeID[cur_id]


def LT_loss(s, c_id, treeID, AL, alpha=1):
    loss1, loss2 = 0, 0
    for i in range(len(c_id)):
        cur_tree_idx = torch.tensor(AL[i], dtype=torch.int64)
        cur_idx = c_id[i]
        loss1 += s[cur_idx, c_id].sum() / len(c_id)

        loss2 += s[cur_idx, cur_tree_idx].sum() / len(AL[i])
    loss = loss1 - loss2 * alpha

    return loss

import torch.nn as nn
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, Sequential, SAGEConv, GATConv
import torch
import time
import random
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon,TUDataset,Coauthor
import torch.nn.functional as F
import torch_geometric.transforms as T
import model.LTLayer_node as lt
from torch_geometric.utils import mask_to_index
from copy import deepcopy
import numpy as np
from torch_geometric.nn import ChebConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

class GNN(nn.Module):
    def __init__(self, conv, feature, hidden, output):
        super(GNN, self).__init__()

        self.input_size, self.output_size = feature, output

        layers = []
        for i in range(len(hidden)):
            if i == 0:
                layers.append((eval(conv)(feature, hidden[i]), 'x, edge_index -> x'), )

            else:
                layers.append((eval(conv)(hidden[i - 1], hidden[i]), 'x, edge_index -> x'), )
            layers.append(nn.ReLU())
            layers.append((nn.Dropout(0.5), 'x -> x'), )

        # layers.append(ChebConv(hidden[i], hidden[i], 2))

        layers.append((eval(conv)(hidden[-1], output), 'x, edge_index -> x'), )

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

class LTTrainer:
    def __init__(self, in_channels, hidden_channels, out_channels, data, dc, k, num_classes, alpha, conv):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dc = dc
        self.k = k
        self.num_classes = num_classes
        self.conv = conv
        self.gcn_embed = GNN(self.conv, self.in_channels, self.hidden_channels, self.hidden_channels[-1])
        self.classifer = nn.Sequential(nn.Linear(self.hidden_channels[-1], 64), nn.Linear(64, self.num_classes))
        self.gcn_pred = GNN(self.conv, data.num_features, hidden_channels, self.num_classes)
        self.lt_layer = lt.LTLayer(self.hidden_channels[-1], self.dc)
        self.data = deepcopy(data).to(device)
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def train(self, model, epochs, name, pre=True, verbose=False):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = F.log_softmax(model(self.data), dim=-1)
            pred = output[self.data.train_mask].max(dim=-1)[1]
            if pre:
                loss = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            else:
                loss = F.nll_loss(output[self.data.running_train_mask], self.data.running_y[self.data.running_train_mask])
            loss.backward()
            optimizer.step()
            if verbose is True:
                print('train, model: {}, epoch: {}, loss: {:.4f}, train acc: {:.4f}'
                      .format(name, epoch, float(loss),
                              (pred == self.data.y[self.data.train_mask]).sum() / self.data.train_mask.sum()))

    # def lt_train(self, epochs):
    #     model = nn.Sequential(self.gcn_embed, self.lt_layer).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    #     pre_loss, cnt = 0, 0
    #     for epoch in range(epochs):
    #         model.train()
    #         optimizer.zero_grad()
    #         s = model(self.data)
    #         c_id, treeID, AL = lt.LTCluster(s, self.data.edge_index, self.k).construct()
    #         # 计算 LT_loss
    #         lt_loss = lt.LT_loss(s, c_id, treeID, AL, self.alpha)
    #         # 计算交叉熵损失
    #         output = self.gcn_embed(self.data)
    #         cross_entropy_loss = self.cross_entropy_loss(output, self.data.y)
    #         # 组合 LT_loss 和交叉熵损失
    #         loss = lt_loss + cross_entropy_loss
    #         # if 0 < pre_loss <= loss or loss <= 0:
    #         #     cnt += 1
    #         # if cnt == 1:
    #         #     break
    #         loss.backward()
    #         optimizer.step()
    #         print('lt_train, epoch: {}, loss: {:.4f}'.format(epoch, float(loss)))
    #         # pre_loss = loss
    # def lt_train(self, epochs):
    #
    #     model = nn.Sequential(self.gcn_embed, self.lt_layer).to(device)
    #
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    #     pre_loss, cnt = 0, 0
    #
    #     # 初始化存储损失的列表
    #     lt_losses = []
    #     cross_entropy_losses = []
    #     total_losses = []
    #
    #     # 设置字体和字号
    #     plt.rcParams['font.family'] = 'Times New Roman'
    #     plt.rcParams['font.size'] = 18
    #
    #     # 创建一个图表
    #     plt.figure(figsize=(10, 8))
    #
    #     for epoch in range(epochs):
    #         model.train()
    #         optimizer.zero_grad()
    #         s = model(self.data)
    #         c_id, treeID, AL = lt.LTCluster(s, self.data.edge_index, self.k).construct()
    #
    #         # 计算 LT_loss
    #         lt_loss = lt.LT_loss(s, c_id, treeID, AL, self.alpha)
    #
    #         # 计算交叉熵损失
    #         output = F.log_softmax(model(self.data), dim=-1)
    #         cross_entropy_loss = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
    #
    #         # 组合 LT_loss 和交叉熵损失
    #         loss = cross_entropy_loss+0.2*lt_loss
    #
    #         print(f'{cross_entropy_loss}')
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 打印当前 epoch 的损失值
    #         print('lt_train, epoch: {}, loss: {:.4f}'.format(epoch, float(loss)))
    #
    #         # 存储每个 epoch 的损失
    #         total_losses.append(float(loss))
    #
    #         # 清空图表，重新绘制
    #         plt.clf()
    #
    #         # 绘制三种损失的变化趋势
    #         plt.plot(range(epoch + 1), total_losses, label='Total Loss')
    #
    #         # 添加图表标题和标签
    #         plt.xlabel('Epoch',fontsize=18)
    #         plt.ylabel('Total Loss',fontsize=18)
    #         # 在图表底部添加注释
    #         plt.figtext(0.5, -0.05, 'λ=0.2', ha='center', va='top', fontsize=18)
    #
    #         # 显示图例
    #         plt.legend()
    #
    #         # 动态更新图表
    #         plt.pause(0.1)
    #
    #     # 保存图表，保存到指定路径
    #     plt.savefig("loss_trends_epoch_{}.png".format(epoch + 1))
    #     print("Epoch {} loss plot saved!".format(epoch + 1))
    #
    #     # 关闭图表
    #     plt.close()
    def lt_train(self, epochs):

        model = nn.Sequential(self.gcn_embed, self.lt_layer).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        pre_loss, cnt = 0, 0

        # 初始化存储损失的列表
        lt_losses = []
        cross_entropy_losses = []
        total_losses = []

        # 设置字体和字号
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # 创建一个图表
        plt.figure(figsize=(10, 8))

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            s = model(self.data)
            c_id, treeID, AL = lt.LTCluster(s, self.data.edge_index, self.k).construct()

            # 计算 LT_loss
            lt_loss = lt.LT_loss(s, c_id, treeID, AL, self.alpha)

            # 计算交叉熵损失
            output = F.log_softmax(model(self.data), dim=-1)
            cross_entropy_loss = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])

            # 组合 LT_loss 和交叉熵损失
            loss = cross_entropy_loss + 0.5 * lt_loss
            print(loss)
            loss.backward()
            optimizer.step()
            # 打印当前 epoch 的损失值
            print('lt_train, epoch: {}, loss: {:.4f}'.format(epoch, float(loss)))

            # # 存储每个 epoch 的损失
            # total_losses.append(float(loss))
            #
            # # 清空图表，重新绘制
            # plt.clf()
            # # 绘制总损失
            # plt.plot(range(epoch + 1), total_losses, label='Total Loss')
            # # 添加图表标题和标签
            # plt.xlabel('Epoch (λ=0.01)', fontsize=18)
            # plt.ylabel('Total Loss', fontsize=18)
            # # 调整布局，确保所有元素都显示完整
            # plt.tight_layout()
            #
            # # 显示图例
            # plt.legend()
            # # 动态更新图表
            # plt.draw()
            # plt.pause(0.1)

        # # 保存图表，保存到指定路径
        # plt.savefig("loss_trends_epoch_{}.png".format(epoch + 1))
        # print("Epoch {} loss plot saved!".format(epoch + 1))
        #
        # # 调整布局，确保所有元素都显示完整
        # plt.tight_layout()
        # # 关闭图表
        # plt.close()

    @torch.no_grad()
    def pseudo_labeling(self, pseudo_label_num=None):
        print(f'Pseudo label num: {pseudo_label_num}')  # 检查伪标签数量是否正确传递
        model = nn.Sequential(self.gcn_embed, self.lt_layer).to(device)
        model.eval()
        s = model(self.data)
        c_id, treeID, AL = lt.LTCluster(s, self.data.edge_index, self.k).construct()
        train_idx = mask_to_index(self.data.train_mask)
        is_not_in_train = c_id[torch.isin(c_id.to(device), train_idx, invert=True)].to(device)

        # 控制生成的伪标签数量
        if pseudo_label_num is not None and pseudo_label_num < is_not_in_train.size(0):
            is_not_in_train = is_not_in_train[:pseudo_label_num]

        print(f'Number of pseudo labels generated: {is_not_in_train.size(0)}')  # 检查最终生成的伪标签数量

        out = F.log_softmax(self.gcn_pred(self.data), dim=-1)
        p, pred = out.max(dim=-1)
        pseudo_labels = pred[is_not_in_train]

        print('pseudo_labels: {}\n, acc: {}/{}'.format(pseudo_labels,
                                                       (pseudo_labels == self.data.y[is_not_in_train]).sum(),
                                                       pseudo_labels.size(0)))

        self.data.running_train_mask = torch.zeros_like(self.data.train_mask)
        self.data.running_train_mask[train_idx] = True
        self.data.running_train_mask[is_not_in_train] = True
        self.data.running_test_mask = deepcopy(self.data.test_mask)
        self.data.running_test_mask[is_not_in_train] = False
        self.data.running_y = deepcopy(self.data.y)
        self.data.running_y[is_not_in_train] = pseudo_labels

        return (pseudo_labels == self.data.y[is_not_in_train]).sum() / pseudo_labels.size(0)

    @torch.no_grad()
    def test(self, model):
        model.to(device)
        model.eval()
        output = F.log_softmax(model(self.data), dim=-1)
        p, pred = output[self.data.running_test_mask].max(dim=-1)
        correct = (pred == self.data.y[self.data.running_test_mask]).sum()
        test_acc = correct / self.data.running_test_mask.sum()
        print('test acc: {:.4f}'.format(test_acc))

        return test_acc

    def process(self, epochs):
        self.train(nn.Sequential(self.gcn_embed, self.classifer), epochs, 'gcn_embed')
        self.train(self.gcn_pred, epochs, 'gcn_pred')
        self.lt_train(50)
        pseudo_acc = self.pseudo_labeling()
        self.train(self.gcn_pred, epochs, 'gcn_pred', pre=False, verbose=False)
        test_acc = self.test(self.gcn_pred)

        return test_acc, pseudo_acc

class ConvTrainer:
    def __init__(self, in_channels, hidden_channels, out_channels, data, num_classes, conv):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.conv = conv
        self.gcn_pred = GNN(self.conv, data.num_features, hidden_channels, self.num_classes)
        self.data = deepcopy(data).to(device)

    def train(self, model, epochs, name, verbose=False):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = F.log_softmax(model(self.data), dim=-1)
            pred = output[self.data.train_mask].max(dim=-1)[1]
            loss = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            if verbose is True:
                print('train, model: {}, epoch: {}, loss: {:.4f}, train acc: {:.4f}'
                      .format(name, epoch, float(loss),
                              (pred == self.data.y[self.data.train_mask]).sum() / self.data.train_mask.sum()))

    @torch.no_grad()
    def test(self, model):
        model.to(device)
        model.eval()
        output = F.log_softmax(model(self.data), dim=-1)
        p, pred = output[self.data.test_mask].max(dim=-1)
        correct = (pred == self.data.y[self.data.test_mask]).sum()
        test_acc = correct / self.data.test_mask.sum()
        print('test acc: {:.4f}'.format(test_acc))

        return test_acc

    def process(self, epochs):
        self.train(self.gcn_pred, epochs, 'gcn_pred', verbose=False)
        test_acc = self.test(self.gcn_pred)

        return test_acc

def run(args):
    seed = args.seed
    set_random_seeds(seed)
    print('seed = ', seed)

    test_acc_all = []
    pseudo_acc_all = []
    i = 0
    while i < args.runs:
        dataset = load_data(args.dataset, args.num_train_per_class, args.test_size)
        data = dataset[0]
        if args.dataset == 'photo':
            data = random_masking(data, args.num_train_per_class, dataset.num_classes, args.test_size)
        if args.dataset == 'Physics':
            data = random_masking(data, args.num_train_per_class, dataset.num_classes, args.test_size)
        if args.dataset == 'CS':
            data = random_masking(data, args.num_train_per_class, dataset.num_classes, args.test_size)
        eta = int(data.num_nodes / (data.num_edges / data.num_nodes) ** (len(args.layers) + 1))
        print('eta: ', eta)

        if args.use_lt is True:
            trainer = LTTrainer(data.num_features, args.layers, dataset.num_classes, data, args.dc, 100, dataset.num_classes, args.alpha, args.conv)
            test_acc, pseudo_acc = trainer.process(args.epochs)  # 这里传递 epochs 参数
            print('test acc in {} run: {:.4f}, pseudo acc: {:.4f}'.format(i, test_acc, pseudo_acc))
            test_acc_all.append(float(test_acc))
            pseudo_acc_all.append(float(pseudo_acc))
        else:
            trainer = ConvTrainer(data.num_features, args.layers, dataset.num_classes, data, dataset.num_classes, args.conv)
            test_acc = trainer.process(args.epochs)  # 这里传递 epochs 参数
            test_acc_all.append(float(test_acc))

        i += 1

    print(test_acc_all)
    if args.use_lt is True:
        print('result in {} runs:\n test acc: {:.4f}\n test std: {:.4f}, pseudo acc: {:.4f}'.format(args.runs,
                                                                                                    np.mean(test_acc_all),
                                                                                                    np.std(test_acc_all),
                                                                                                    np.mean(pseudo_acc_all)))
    else:
        print('result in {} runs:\n test acc: {:.4f}\n test std: {:.4f}'.format(args.runs, np.mean(test_acc_all), np.std(test_acc_all)))

def load_data(dataset, num_train_per_class, num_test):
    if dataset == 'photo':
        return Amazon(root='./data/{}'.format(dataset), name=dataset, transform=None)
    if dataset == 'Physics':
        return Coauthor(root='./data/{}'.format(dataset), name=dataset, transform=None)
    if dataset == 'CS':
        return Coauthor(root='./data/{}'.format(dataset), name=dataset, transform=None)
    else:
        return Planetoid(root='./data/{}'.format(dataset), name=dataset, split="random",
                         num_train_per_class=num_train_per_class,
                         num_val=0, num_test=num_test)

def random_masking(data, num_train_per_class, num_classes, num_test):
    train_mask = torch.zeros(size=(data.num_nodes,), dtype=torch.bool)
    train_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    test_mask = torch.zeros(size=(data.num_nodes,), dtype=torch.bool)
    test_mask.fill_(False)
    test_mask[remaining[:num_test]] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data

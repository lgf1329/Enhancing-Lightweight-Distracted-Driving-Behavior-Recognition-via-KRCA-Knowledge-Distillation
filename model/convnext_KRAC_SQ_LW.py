#数据库的导入

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import copy
import time
import random
import os
from PIL import Image
from IPython.display import display
from torchvision import models

import torchvision
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ghostnet.crcd.criterion import CRCDLoss

# 学生网络 Goast
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from ghostnet.distillers.DKD import dkd_loss
from ghostnet.Convnet import  convnext_tiny
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import warnings

import gc
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from ptflops import get_model_complexity_info
from SqueezeNet.model_SQ import SqueezeNet1
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="size_average and reduce args will be deprecated")

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)  # parameters的封装使得变量可以容易访问到

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum = max(loss_sum, 0)
        # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum



SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data_dir_train = 'E:\\Code fusion\\un-20usatrain_no_noise\\'
# data_dir_test = 'E:\\Code fusion\\un-6usatest_no_noise\\'
# labels_train = pd.read_csv('E:\\Code fusion\\list_20usatrain_no_noise-new.csv')
# labels_test = pd.read_csv('E:\\Code fusion\\list_6usatest_no_noise.csv')
'---------------SF数据集---------------------'
data_dir_train = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/train'
data_dir_test = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/test'
labels_train = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/train_labels.csv')
labels_test = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/test_labels.csv')
save_model_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/ghostnet/模型保存_Conv-SQ-KRCA-13"
teacher_model_weight_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/convnext_集合后重新划分/convnext50.pth"
# model_weight_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/SqueezeNet_集合后重新划分/SqueezeNet85.pth"
model_weight_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/SqueezeNet_集合后重新划分/SqueezeNet13.pth"
# '---------------ASU数据集---------------------'
# data_dir_train = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/train'
# data_dir_test = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/test'
# labels_train = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/train_labels.csv')
# labels_test = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/test_labels.csv')
# save_model_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/ghostnet_ASU/模型保存_Conv-SQ-KRCA_NOHD5"
# # save_model_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/ghostnet_ASU/模型保存_Conv-SQ-KRCA"
# teacher_model_weight_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/convnext_ASU_集合后重新划分/convnext14.pth"
# model_weight_path = r"/home/luo/知识蒸馏/4-代码/第三章/模型参数/SqueezeNet_ASU_集合后重新划分/SqueezeNet46.pth"


BATCH_SIZE = 4  # 从1调整为4，配合梯度累积
ACCUM_STEPS = 16  # 梯度累积步数

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(f'{images[i][1]}')
        ax.imshow(np.array(images[i][0]))
        ax.axis('off')



class CIFAR100InstanceSample():
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self, train_data,
                 transform=None, target_transform=None,
                 download=False, k=512, mode='exact', is_sample=True, percent=1.0):  # 4096
        super().__init__()
        self.k = k  # number of negative samples for NCE
        self.mode = mode
        self.is_sample = is_sample
        self.train_data = train_data
        self.transform = transform
        self.target_transform = target_transform
        num_classes = 10
        #datas = []
        labels = []
        # data, labels = train_data
        for _, label in [train_data[i] for i in range(len(train_data))]:
            #datas.append(data)
            labels.append(label)

        num_samples = len(train_data)
        # print(num_samples)
        label = labels
        # num_samples = len(self.test_data)
        # label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]  #
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.random.choice(np.array(self.cls_positive[i]), 1900, replace=True) for i in
                             range(num_classes)]
        self.cls_negative = [np.random.choice(np.array(self.cls_negative[i]), 17100, replace=True) for i in
                             range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)  #
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.array(self.cls_positive)  # 100, 500
        self.cls_negative = np.array(self.cls_negative)  # 100, 49500
        if self.mode == "queue":
            import queue
            # sample_idx = np.random.choice(np.arange(num_samples), self.k + 64, replace=False) #
            # self.sample_idx_queue = queue.Queue(maxsize=self.k + 64)
            # for i in sample_idx:
            #     self.sample_idx_queue.put(i)
            self.bs = 64
            self.sample_idx = np.random.choice(np.random.choice(np.arange(num_samples), 10000, replace=False),
                                               [self.k + 64], replace=False)
            self.ptr1 = 0
            self.ptr2 = self.k
            print("the len of queue is {}".format(len(list(self.sample_idx))))

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index):
        index_tem = index

        img, target = self.train_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)  # choose a positive
                pos_idx = pos_idx[0]
            elif self.mode == 'queue':
                pos_idx = [index]  # choose a positive
                if self.ptr2 > self.ptr1:
                    neg_idx = self.sample_idx[self.ptr1:self.ptr2].copy()
                else:
                    neg_idx = np.concatenate((self.sample_idx[:self.ptr2], self.sample_idx[self.ptr1:])).copy()
                self.sample_idx[self.ptr1] = index
                assert neg_idx.shape[0] == self.k
                self.ptr1 += 1
                self.ptr2 += 1
                if self.ptr1 >= self.k + self.bs:
                    self.ptr1 = 0
                if self.ptr2 > self.k + self.bs:
                    self.ptr2 = 1
                sample_idx = np.hstack((np.asarray(pos_idx), neg_idx))
                # if self.sample_idx_queue.full():
                #     self.sample_idx_queue.get()
                # self.sample_idx_queue.put(index, False)
                return img, target, index, sample_idx
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
#
# # 优化后的分层数据加载
# def create_stratified_loader(data_dir, transform, batch_size, ratio=1.0):
#     """创建分层抽样的DataLoader"""
#     full_dataset = torchvision.datasets.ImageFolder(root=data_dir)
#
#     if ratio < 1.0:
#         # 分层抽样逻辑
#         targets = np.array(full_dataset.targets)
#         class_indices = [np.where(targets == i)[0] for i in np.unique(targets)]
#
#         subset_indices = []
#         for indices in class_indices:
#             n_samples = int(len(indices) * ratio)
#             subset_indices.extend(np.random.choice(indices, n_samples, replace=False))
#
#         np.random.shuffle(subset_indices)
#         dataset = Subset(full_dataset, subset_indices)
#     else:
#         dataset = full_dataset  # 使用完整数据集
#
#     # 应用数据增强
#     if ratio < 1.0:
#         dataset.dataset.transform = transform
#     else:
#         dataset.transform = transform
#
#     return DataLoader(
#         CIFAR100InstanceSample(dataset, transform=transform),
#         batch_size=batch_size,
#         shuffle=True if ratio == 1.0 else False,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True
#     )



# ========== 修改3：修正数据加载器创建逻辑 ==========
def create_stratified_loader(data_dir, transform, batch_size, ratio=1.0):
    # 创建基础数据集（禁用默认transform）
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=None  # 关键修改：不在父数据集应用transform
    )

    # 保持原有的分层抽样逻辑不变
    if ratio < 1.0:
        targets = np.array([s[1] for s in full_dataset.samples])
        class_indices = [np.where(targets == i)[0] for i in np.unique(targets)]

        subset_indices = []
        for indices in class_indices:
            n_samples = int(len(indices) * ratio)
            subset_indices.extend(np.random.choice(indices, n_samples, replace=False))

        np.random.shuffle(subset_indices)
        dataset = Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset

    # 创建增强数据集时统一应用transform
    return DataLoader(
        CIFAR100InstanceSample(
            dataset, transform=transform,
            download=False,
            k=2048,
            mode='exact',
            is_sample=True,
            percent=1.0
        ),
        batch_size=batch_size,
        shuffle=True if ratio == 1.0 else False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    #acc = correct.float() / y.shape[0]
    return correct.float() / y.shape[0]

def get_losses_weights(losses):
	if type(losses) != torch.Tensor:
	    losses = torch.tensor(losses)
	weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
	return weights

#加了注意力的 128
#def distillation(y, labels, teacher_scores, s_feat, t_feat, temp, alpha):
    #return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1))* (
           #temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) *(1-alpha)
def distillation(y, labels, teacher_scores, temp, alpha):
    return  F.cross_entropy(y, labels) * alpha#(1-alpha)  #nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1))* (temp * temp * 2.0 * alpha),
def AFe(s_feat, t_feat, temp, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(s_feat / temp, dim=1), F.softmax(t_feat / temp, dim=1)) * ( temp * temp * 2.0 * alpha)

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, emd_fc_type='linear'):
        super(Embed, self).__init__()
        if emd_fc_type == "linear":
            self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print('x', x.shape)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SpatialAttention2d(nn.Module):
    def __init__(self, dim_in):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Linear(dim_in, dim_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z, z


class CRCDFE(nn.Module):
    """CRCD Loss function

    Args:
        opt.embed_type: fc;nofc;nonlinear
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, s_dim, t_dim, feat_dim=128, embed_type='linear'):
        super(CRCDFE, self).__init__()
        self.emd_fc_type = embed_type
        # print("fc_type: {} ".format(self.emd_fc_type))
        if self.emd_fc_type == "nofc":
            assert s_dim == t_dim
            feat_dim = s_dim
        self.embed_s = Embed(s_dim, feat_dim, self.emd_fc_type)
        self.embed_t = Embed(t_dim, feat_dim, self.emd_fc_type)
        self.SpatiaAttention_s = SpatialAttention2d(feat_dim)
        self.SpatiaAttention_t = SpatialAttention2d(feat_dim)

    def forward(self, f_s, f_t):
        """
        There may be some learnable parameters in embedding layer in teacher side,
        similar to crd, we also calculate the crcd loss over both the teacher and the student side.
        However, if the emd_fc_type=="nofc", then the 't_loss' term can be omitted.
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        # print('f_sss', f_s.is_cuda)
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        f_s,C_s = self.SpatiaAttention_s(f_s)
        f_t,C_t = self.SpatiaAttention_t(f_t)
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s)
        return mask_loss #f_s, f_t


def train_student_kd(model, device, train_loader, optimizer, awl1, awl2, criterion_kd_grad, criterion_FE,epoch):
    model.train()
    trained_samples = 0
    n_data=len(train_loader)
    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 =0
    epoch_mask_loss = 0
    epoch_acc = 0
    sigmoid = nn.Sigmoid()
    # Metrics lists for precision, recall, and F1 calculation across all batches
    all_targets = []
    all_preds = []
    # Start time
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        img, target, index, contrast_idx = data
        # print('contrast_idx.shape',contrast_idx.shape)
        img = img.float()
        img = img.to(device)
        target = target.to(device)
        index = index.to(device)
        contrast_idx = contrast_idx.to(device)

        optimizer.zero_grad()
        feat_s1, feat_s2, output = model(img)
        # print(f"初始卷积输出形状: {feat_s1.shape}")  # 应输出 torch.Size([4, 16, 112, 112])
        # print(f"特征层输出形状: {feat_s2.shape}")  # 应输出 torch.Size([4, 960, 7, 7])
        # print(f"分类器输出形状: {output.shape}")  # 应输出 torch.Size([4, 10])
        feat_t1, feat_t2, teacher_output = teacher_model(img)
        # print(f"初始卷积输出形状: {feat_t1.shape}")  # 应输出 torch.Size([4, 16, 56, 56])
        # print(f"特征层输出形状: {feat_t2.shape}")  # 应输出 torch.Size([4, 768, 7, 7])
        # print(f"分类器输出形状: {teacher_output.shape}")  # 应输出 torch.Size([4, 10])
        teacher_output = teacher_output.detach()  # 切断老师网络的反向传播，感谢B站“淡淡的落”的提醒
        feat_t1 = feat_t1.detach()
        feat_t2 = feat_t2.detach()

        feat_s1 = feat_s1.view(feat_s1.size(0), -1)
        feat_t1 = feat_t1.view(feat_t1.size(0), -1)
        feat_s2 = feat_s2.view(feat_s2.size(0), -1)
        feat_t2 = feat_t2.view(feat_t2.size(0), -1)

        f_s = feat_s2
        f_t = feat_t2

        loss1, mask_loss = criterion_kd_grad(f_s, f_t, index, contrast_idx)
        # print(f"loss1: {loss1}")  # 应输出 torch.Size([4, 10])
        #s_feat, t_feat = criterion_FE(feat_s1, feat_t1)
        #s_feat, t_feat = criterion_FE(feat_s1, feat_t1)
        loss3 = distillation(output, target, teacher_output, temp=5.0, alpha=0.7)
        loss2 = dkd_loss(
            output,
            teacher_output,
            target,
            1,
            8,
            4,
        )
        loss4 = criterion_FE(feat_s1, feat_t1)

        loss = awl1(loss1, loss2, loss3) + 0.5*awl2(loss4 + mask_loss)

        acc = calculate_accuracy(output, target)

        # Append predictions and targets for later metrics calculation
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss1 += loss1.item()
        epoch_loss2 += loss2.item()
        epoch_loss3 += loss3.item()
        epoch_loss4 += loss4.item()
        epoch_mask_loss += mask_loss.item()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        trained_samples += len(img)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

    # Calculate precision, recall, F1 scores and end time
        training_time = time.time() - start_time
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')


        # Calculate per-sample averages for losses and accuracy
        avg_loss1 = epoch_loss1 / n_data
        avg_loss2 = epoch_loss2 / n_data
        avg_loss3 = epoch_loss3 / n_data
        avg_loss4 = epoch_loss4 / n_data
        avg_mask_loss = epoch_mask_loss / n_data
        avg_total_loss = epoch_loss / n_data
        avg_acc = epoch_acc / len(train_loader)

    return avg_loss1, avg_loss2, avg_loss3, avg_loss4, avg_mask_loss, avg_total_loss, avg_acc, precision, recall, f1, training_time

def test_student_kd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # Metrics lists to collect predictions and targets
    all_targets = []
    all_preds = []

    # Start time
    start_time = time.time()
    # 创建 CrossEntropyLoss 实例并指定 reduction='sum'
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feat_s1, feat_s2, output = model(data)
            test_loss += criterion(output, target).item()  # 计算损失
            warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn')
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Append predictions and targets for metric calculation
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    test_time = time.time() - start_time
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')



    # Calculate accuracy
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy, precision, recall, f1, test_time



num_training_examples = 0
for fol in os.listdir(data_dir_train):
    num_training_examples += len(os.listdir(os.path.join(data_dir_train, fol)))
num_test_examples = 0
for fol in os.listdir(data_dir_test):
    num_test_examples += len(os.listdir(os.path.join(data_dir_test, fol)))




def count_parameters(model):
    # 计算输入为224x224x3时的FLOPs和参数数量
    with torch.cuda.device(0):  # 如果有GPU可用，选择GPU计算
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    return flops,params

def student_kd_main(save_model_path):
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化混合精度训练组件[3,7](@ref)
    scaler = GradScaler()

    # 模型加载优化
    model = SqueezeNet1(version='custom',num_classes=10)
    model.load_state_dict(torch.load(model_weight_path))
    # model.classifier = nn.Linear(model.classifier.in_features, 10)
    model = model.to(device)
    flops, params = count_parameters(model)

    print(f"FLOPs: {flops}")
    print(f"参数数量 (Weights): {params}")
    # 教师模型冻结优化[2](@ref)
    teacher_model = convnext_tiny()
    teacher_model.head = nn.Linear(teacher_model.head.in_features, 10)
    teacher_model.load_state_dict(torch.load(teacher_model_weight_path))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 设置为评估模式
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 损失函数组件内存优化
    awl1 = AutomaticWeightedLoss(3).to(device)
    awl2 = AutomaticWeightedLoss(2).to(device)
    criterion_kd_grad = CRCDLoss(6272, 37632, 64, 472080, nce_k=2048).to(device)
    criterion_FE = CRCDFE(1140576, 301056, 64).to(device)

    # 优化器配置优化[5](@ref)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': awl1.parameters(), 'weight_decay': 0},
        {'params': awl2.parameters(), 'weight_decay': 0},
        {'params': criterion_kd_grad.parameters()},  # 合并损失函数参数
        {'params': criterion_FE.parameters()}
    ],
    lr=0.002,          # 学习率（推荐初始值）
    weight_decay=0.01, # 权重衰减（推荐值）
        )#Adadelta

    # 训练历史记录
    student_history = []
    best_loss = 100
    best_score = 0
    data_fenges = 10
    for epoch in range(1, epochs + 1):

        start_time = time.time()
        train_loss1, train_loss2, train_loss3, train_loss4, mask_loss, train_loss, train_acc, train_precision, train_recall, train_f1, train_training_time = train_student_kd(
            model, device, train_loader, optimizer, awl1, awl2, criterion_kd_grad, criterion_FE, epoch)

        loss, acc, precision, recall, f1, test_time = test_student_kd(model, device, test_loader)

        student_history.append((loss, acc))

        if loss < best_loss or acc > best_score:
            if loss < best_loss:
                best_loss = loss
            if acc > best_score:
                best_score = acc
            torch.save(model.state_dict(),
                       os.path.join(save_model_path, 'SqueezeNet{}.pth'.format(epoch + 1)))  # save spatial_encoder
            # print("Epoch {} model saved!".format(epoch))

        # 内存最终清理
        torch.cuda.empty_cache()
        gc.collect()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {epoch :02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss1: {train_loss1:.3f} | Train Loss2: {train_loss2:.3f} | Train Loss3: {train_loss3:.3f} | Train Loss4: {train_loss4:.3f} | Train Loss5: {mask_loss:.3f} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%| train_precision: {train_precision * 100:.2f}%| train_recall: {train_recall* 100:.2f}%| train_f1: {train_f1  * 100:.2f}%| train_training_time: {train_training_time :.2f}s')
        # print(f'\t Val. Loss: {loss:.3f} |  Val. Acc: {acc * 100:.2f}%|  Val. precision: {precision * 100:.2f}%|  Val. recall: {recall * 100:.2f}%|  Val. f1: {f1 * 100:.2f}%|  Val. test_time: {test_time:.2f}')
        print(f'{epoch :02} | {epoch_mins}m {epoch_secs}s')
        print(
            f'T{train_loss1:.3f} | {train_loss2:.3f} | {train_loss3:.3f} | {train_loss4:.3f} |  {mask_loss:.3f} |  {train_loss:.3f} | {train_acc * 100:.2f}%|  {train_precision * 100:.2f}%|  {train_recall * 100:.2f}%| {train_f1 * 100:.2f}%|  {train_training_time :.2f}s')
        print(
            f' {loss:.3f} |  {acc * 100:.2f}%|  {precision * 100:.2f}%| {recall * 100:.2f}%| {f1 * 100:.2f}%|  {test_time:.2f}s')

    # torch.save(model.state_dict(), "student_kd_mo_gh.pt")
    return model, student_history




# 数据集加载优化：分块加载
train_data = torchvision.datasets.ImageFolder(root=data_dir_train)
test_data = torchvision.datasets.ImageFolder(root=data_dir_test)
# 数据增强优化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)



# # ========== 修改1：调整数据增强中的Lambda转换 ==========
# train_transforms = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),  # 添加类型检查
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize
# ])
#
# test_transforms = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),  # 添加类型检查
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize
# ])
BATCH_SIZE = 4  # 从1调整为4，配合梯度累积
ACCUM_STEPS = 16  # 梯度累积步数

train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # 统一图像模式
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 使用子集数据
train_set = CIFAR100InstanceSample(
    train_data,  # 替换原来的train_data
    download=False,
    transform=train_transforms,
    k=2048,
    mode='exact',
    is_sample=True,
    percent=1.0
)

# DataLoader优化配置
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=10,  # 避免过多worker导致内存溢出
    pin_memory=True,
    persistent_workers=True
)

valid_data = copy.deepcopy(test_data)
valid_data.transform = test_transforms
test_loader = DataLoader(
    valid_data,
    shuffle=True,
    batch_size=BATCH_SIZE * 2
)

# 教师模型加载优化
teacher_model = convnext_tiny()
teacher_model.head = nn.Linear(teacher_model.head.in_features, 10)
teacher_model.load_state_dict(torch.load(teacher_model_weight_path))
teacher_model = teacher_model.to(device)
teacher_model.eval()  # 冻结教师模型
for param in teacher_model.parameters():
    param.requires_grad = False



# 启动训练
student_kd_model, student_kd_history = student_kd_main(save_model_path)

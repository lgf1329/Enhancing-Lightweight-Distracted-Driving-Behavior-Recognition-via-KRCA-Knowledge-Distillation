# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author: moyi hang time:2021/7/4

# from torchvision.models import resnet18
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
from ptflops import get_model_complexity_info
import torch
import torchvision
from torch import nn, optim
from torchvision import models
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
plt.style.use('seaborn-whitegrid')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.size'] = 12

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'---------------SF数据集---------------------'
# data_dir_train = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/train'
# data_dir_test = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/test'
# labels_train = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/train_labels.csv')
# labels_test = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_usa_集合后重新划分/test_labels.csv')
# save_model_path = r'/home/luo/知识蒸馏/4-代码/第三章/模型参数/原始SqueezeNet_集合后重新划分'#SF数据集存放地址
'---------------ASU数据集---------------------'
data_dir_train = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/train'
data_dir_test = r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/test'
labels_train = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/train_labels.csv')
labels_test = pd.read_csv(r'/home/luo/知识蒸馏/4-代码/第三章/数据/data_ASU_all_集合后重新划分/test_labels.csv')
save_model_path = r'/home/luo/知识蒸馏/4-代码/第三章/模型参数/原始SqueezeNet_ASU_集合后重新划分'#SF数据集存放地址


class DrivingDataset():
    def __init__(self, data_dir, img_size=224):
        self.data_dir = Path(data_dir)

        # Step 1. 自动发现类别目录
        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        assert len(self.class_dirs) > 0, f"数据集目录 {data_dir} 中没有找到有效的类别子目录"

        # Step 2. 递归收集所有图片路径（路径字符串，类别名称）
        self.img_paths = []
        for class_dir in self.class_dirs:
            class_name = class_dir.name
            # 递归搜索所有层级的图片文件
            img_files = list(class_dir.glob("**/*.jpg")) + list(class_dir.glob("**/*.png"))
            self.img_paths += [
                (str(img_path.absolute()), class_name)
                for img_path in img_files
            ]

        # Step 3. 验证数据有效性
        assert len(self.img_paths) > 0, f"数据集目录 {data_dir} 中未找到任何图片"
        print(f"成功加载 {len(self.img_paths)} 张图片，类别分布：")
        for cls in set(c for _, c in self.img_paths):
            print(f"  {cls}: {sum(1 for _, c in self.img_paths if c == cls)} 张")

        # Step 4. 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Step 5. 建立类别映射
        self.classes = sorted(set(c for _, c in self.img_paths))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"类别映射: {self.class_to_idx}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.img_paths[idx]  # 这里已确保解包安全

        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            # 转换标签
            label = self.class_to_idx[class_name]
        except Exception as e:
            print(f"处理图片失败: {img_path} | 错误: {str(e)}")
            # 返回空数据防止中断
            return torch.zeros(3, 224, 224), 0

        return self.transform(img), label

BATCH_SIZE = 64

train_iterator = DataLoader(DrivingDataset(data_dir_train),
                            shuffle=True,
                            batch_size=BATCH_SIZE)

valid_iterator = DataLoader(DrivingDataset(data_dir_test),
                            shuffle=True,
                            batch_size=BATCH_SIZE)

def count_parameters(model):
    # 计算输入为224x224x3时的FLOPs和参数数量
    with torch.cuda.device(0):  # 如果有GPU可用，选择GPU计算
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    return flops, params


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = Variable(torch.FloatTensor(np.array(x))).to(device)
        y = Variable(torch.LongTensor(y)).to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = Variable(torch.FloatTensor(np.array(x))).to(device)
            y = Variable(torch.LongTensor(y)).to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):
    """ Fits a dataset to model"""
    best_valid_loss = float('inf')
    best_valid_acc = 0

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    # test_losses = []
    # test_accs = []
    for epoch in range(epochs):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, loss_criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss_criterion, device)
        test_fps()
        # test_lossd, test_acc = evaluate(model, test_iterator, loss_criterion, device)
        # if epoch < 10:
        # torch.save(model.state_dict(), os.path.join(save_model_path, 'mobilenet{}.pth'.format(epoch + 1)))  # save spatial_encoder
        # if epoch > 20:
        # torch.save(model.state_dict(), os.path.join(save_model_path, 'mobilenet{}.pth'.format(epoch + 1)))  # save spatial_encoder

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # test_losses.append(test_lossd)
        train_accs.append(train_acc * 100)
        valid_accs.append(valid_acc * 100)
        # test_accs.append(test_acc * 100)
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),
                       os.path.join(save_model_path, 'sqnet{}.pth'.format(epoch + 1)))  # save spatial_encoder
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save_model_path, 'sqnet{}.pth'.format(epoch + 1)))

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'{train_loss:.3f} | {train_acc * 100:.2f}%  |  {valid_loss:.3f} | {valid_acc * 100:.2f}%')
        # print(f'\t test. Loss: {test_lossd:.3f} |  test. Acc: {test_acc * 100:.2f}%')

    return pd.DataFrame({f'{model_name}_Training_Loss': train_losses,
                         f'{model_name}_Training_Acc': train_accs,
                         f'{model_name}_Validation_Loss': valid_losses,
                         f'{model_name}_Validation_Acc': valid_accs})
    # f'{model_name}_test_Loss': test_losses,
    # f'{model_name}_test_Acc': test_accs,})


def plot_training_statistics(train_stats, model_name):
    fig, axes = plt.subplots(2, figsize=(15, 15))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label=f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label=f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label=f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label=f'{model_name}_Validation_Acc')

    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")

    axes[0].legend(), axes[1].legend()

def test_fps(batch_size=64, n_warmup=10, n_test=100):
    """测试模型推理速度(FPS)"""
    # 准备测试数据
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    # 预热
    print("Warming up...")
    for _ in range(n_warmup):
        _ = model(dummy_input)

    # 基准测试
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    total_time = time.time() - start_time

    # 计算FPS
    fps = n_test / total_time
    print(f"FPS: {fps:.2f} (batch_size={batch_size})")
    return fps


# model = ResNet(num_blocks=[2, 2, 2, 2], num_classes=10)
# model = models.mobilenet_v3_large(pretrained=True)
model = models.squeezenet1_0(pretrained=True)


# 2. 修改分类器
num_classes = 10
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

# 3. 移除 Dropout
model.classifier[0] = nn.Identity()

# 4. 添加展平层（可选）
model.classifier.add_module("flatten", nn.Flatten())
print(model)



model = model.to(device)
loss_criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.Adam(list(model.features[16].parameters()) + list(model.classifier.parameters()), lr = 1e-4 )  #, momentum = 0.9)
test_fps()

flops, params = count_parameters(model)

print(f"FLOPs: {flops}")
print(f"参数数量 (Weights): {params}")
# print(f'The model has {count_parameters(model.features[16]) + count_parameters(model.classifier):,} trainable parameters')
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

# save_path = './MobileNetV3_usa_agument(最后一层全连接).pth'
train_stats_MobileNetV3 = fit_model(model, 'SqueezeNet', train_iterator, valid_iterator, optimizer, loss_criterion,
                                    device,
                                    epochs=100)
# torch.save(model.state_dict(), save_path)
# 标号002
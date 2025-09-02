# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author: moyi hang time:2021/7/4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import random
import os
from torch import nn, optim
import torch
import torchvision
from ptflops import get_model_complexity_info
print(torchvision.__path__)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from SqueezeNet.model_SQ import SqueezeNet1
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
'---------------SFD3---------------------'
# data_dir_train = r'/home/luo/data/SFD3/train'
# data_dir_test = r'/home/luo/data/SFD3/test'
# labels_train = pd.read_csv(r'/home/luo/data/SFD3/train_labels.csv')
# labels_test = pd.read_csv(r'/home/luo/data/SFD3/test_labels.csv')
# save_model_path = r'/home/luo/data/SFD3/model_pth/convnext'
# '---------------AUC---------------------'
data_dir_train = r'/home/luo/data/AUC/train'
data_dir_test = r'/home/luo/data/AUC/test'
labels_train = pd.read_csv(r'/home/luo/data/AUC/train_labels.csv')
labels_test = pd.read_csv(r'/home/luo/data/AUC/test_labels.csv')
save_model_path = r'/home/luo/data/SFD3/model_pth/convnext_AUC'




class DrivingDataset():
    def __init__(self, data_dir, img_size=224):
        self.data_dir = Path(data_dir)


        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]


        self.img_paths = []
        for class_dir in self.class_dirs:
            class_name = class_dir.name

            img_files = list(class_dir.glob("**/*.jpg")) + list(class_dir.glob("**/*.png"))
            self.img_paths += [
                (str(img_path.absolute()), class_name)
                for img_path in img_files
            ]


        for cls in set(c for _, c in self.img_paths):
            print(f"  {cls}: {sum(1 for _, c in self.img_paths if c == cls)} ")


        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.classes = sorted(set(c for _, c in self.img_paths))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.img_paths[idx]

        try:

            img = Image.open(img_path).convert('RGB')

            label = self.class_to_idx[class_name]
        except Exception as e:

            return torch.zeros(3, 224, 224), 0

        return self.transform(img), label

BATCH_SIZE = 64

train_iterator = DataLoader(DrivingDataset(data_dir_train),
                            shuffle=True,
                            batch_size=BATCH_SIZE,
                            num_workers=10)

valid_iterator = DataLoader(DrivingDataset(data_dir_test),
                            shuffle=True,
                            batch_size=BATCH_SIZE,
                            num_workers=10)

def plot_images(images):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(f'{images[i][1]}')
        ax.imshow(np.array(images[i][0]))
        ax.axis('off')




def count_parameters(model):
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    return flops,params


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

        a,b,y_pred = model(x)

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

            a,b,y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):
    """ Fits a dataset to model"""
    print(epochs)
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
        # print('train is ok')
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss_criterion, device)


        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # test_losses.append(test_lossd)
        train_accs.append(train_acc * 100)
        valid_accs.append(valid_acc * 100)
        # test_accs.append(test_acc * 100)
        torch.save(model.state_dict(),
                   os.path.join(save_model_path, 'SqueezeNet{}.pth'.format(epoch + 1)))  # save spatial_encoder
        # if valid_loss <= best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(),
        #                os.path.join(save_model_path, 'SqueezeNet{}.pth'.format(epoch + 1)))  # save spatial_encoder
        # if valid_acc > best_valid_acc:
        #     best_valid_acc = valid_acc
        #     torch.save(model.state_dict(), os.path.join(save_model_path, 'SqueezeNet{}.pth'.format(epoch + 1)))

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'{train_loss:.3f} | {train_acc * 100:.2f}%  |  {valid_loss:.3f} | {valid_acc * 100:.2f}%')
        # print(f'{valid_loss:.3f} |  {valid_acc * 100:.2f}%')

    return pd.DataFrame({f'{model_name}_Training_Loss': train_losses,
                         f'{model_name}_Training_Acc': train_accs,
                         f'{model_name}_Validation_Loss': valid_losses,
                         f'{model_name}_Validation_Acc': valid_accs})



def plot_training_statistics(train_stats, model_name):
    fig, axes = plt.subplots(2, figsize=(15, 15))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label=f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label=f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label=f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label=f'{model_name}_Validation_Acc')

    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")

    axes[0].legend(), axes[1].legend()



model = SqueezeNet1(version='custom',num_classes=10)
# model = SqueezeNet.model_SQ_all.SqueezeNet1(version='custom', num_classes=10)
# model = SqueezeNet.model_SQ_change_tezhen.SqueezeNet1(version='custom', num_classes=10)
# model = SqueezeNet.model_SQ_change_fenlei.SqueezeNet1(version='custom', num_classes=10)
model = model.to(device)
loss_criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=0.002,
#     weight_decay=0.01,
# )

flops, params = count_parameters(model)

print(f"FLOPs: {flops}")
print(f"Weights: {params}")

# if hasattr(torch.cuda, 'empty_cache'):
#     torch.cuda.empty_cache()
#
train_stats_Squeezenet = fit_model(model, 'Squeezenet', train_iterator, valid_iterator, optimizer, loss_criterion,
                                    device,
                                    epochs=100)
# # 标号002
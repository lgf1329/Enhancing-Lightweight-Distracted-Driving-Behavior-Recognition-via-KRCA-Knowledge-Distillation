# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author: moyi hang time:2021/7/4

from torchvision.models import resnet18
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

import torch
import torchvision
from ptflops import get_model_complexity_info

print(torchvision.__path__)
from torch import nn, optim
from torchvision import models
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from ghostnet.Convnet import convnext_tiny
from ghostnet.Convnet import convnext_base
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
# data_dir_train = r'/home/luo/data/SFD3/train'
# data_dir_test = r'/home/luo/data/SFD3/test'
# labels_train = pd.read_csv(r'/home/luo/data/SFD3/train_labels.csv')
# labels_test = pd.read_csv(r'/home/luo/data/SFD3/test_labels.csv')
# save_model_path = r'/home/luo/data/SFD3/model_pth/convnext'
# '---------------ASU数据集---------------------'
data_dir_train = r'/home/luo/data/AUC/train'
data_dir_test = r'/home/luo/data/AUC/test'
labels_train = pd.read_csv(r'/home/luo/data/AUC/train_labels.csv')
labels_test = pd.read_csv(r'/home/luo/data/AUC/test_labels.csv')
save_model_path = r'/home/luo/data/SFD3/model_pth/convnext_AUC'



# display(labels_train.head())
# display(labels_vail.head())
# display(sample_sub.head())

# train_img_dir = os.path.join(data_dir, 'imgs/train')
# test_img_dir = os.path.join(data_dir, 'imgs/test')

num_training_examples = 0
for fol in os.listdir(data_dir_train):
    num_training_examples += len(os.listdir(os.path.join(data_dir_train, fol)))
num_test_examples = 0
for fol in os.listdir(data_dir_test):
    num_test_examples += len(os.listdir(os.path.join(data_dir_test, fol)))

# assert(num_training_examples == len(labels))

# assert(len(os.listdir(test_img_dir)) == len(sample_sub))

train_data = torchvision.datasets.ImageFolder(root=data_dir_train)
test_data = torchvision.datasets.ImageFolder(root=data_dir_test)
labels_train.classname.map(train_data.class_to_idx)
labels_test.classname.map(test_data.class_to_idx)


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


N_IMAGES = 9

images = [(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]]
plot_images(images)

# VALID_RATIO = 0.9
# valid_ratio = 0.1
# n_train_examples = int(len(train_data) * VALID_RATIO)
# n_valid_examples = int(len(train_data)*valid_ratio)
# n_valid_examples = len(train_data) - n_train_examples

# train_data,vail_data = torch.utils.data.random_split(train_data,[n_train_examples,n_valid_examples]
# )
# vail_data = torch.utils.data.random_split(vail_data
# )
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize])
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
train_data.transform = train_transforms
valid_data = copy.deepcopy(test_data)
valid_data.transform = test_transforms
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')

BATCH_SIZE = 32

train_iterator = DataLoader(train_data,
                            shuffle=True,
                            batch_size=BATCH_SIZE)

valid_iterator = DataLoader(valid_data,
                            shuffle=True,
                            batch_size=BATCH_SIZE * 2)


# test_data.transform = test_transforms
# test_iterator = DataLoader(test_data,
# shuffle = True,
# batch_size = BATCH_SIZE*2)#,
# batch_size = BATCH_SIZE)

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
        # print('evaluate is ok')
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
                       os.path.join(save_model_path, 'convnext{}.pth'.format(epoch + 1)))  # save spatial_encoder
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save_model_path, 'convnext{}.pth'.format(epoch + 1)))

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


# model = ResNet(num_blocks=[2, 2, 2, 2], num_classes=10)
model = convnext_tiny(pretrained=True)
# print(model)
# model.classifier = nn.Linear(model.classifier.in_features, 10)

# for name, param in model.named_parameters():
# if("5" not in name):
# param.requires_grad = False

# model.classifier = nn.Sequential(nn.Linear(960, 10))
model.head = nn.Linear(model.head.in_features, 10)
print(model)
# model.classifier[3] = nn.Sequential(
# nn.Linear(960, 1280),
# nn.Hardswish(inplace=True),
# nn.Dropout(p=0.2, inplace=True),
# nn.Linear(1280, 10))

model = model.to(device)
loss_criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.Adam(list(model.features[16].parameters()) + list(model.classifier.parameters()), lr = 1e-4 )  #, momentum = 0.9)

flops, params = count_parameters(model)

print(f"FLOPs: {flops}")
print(f"Weights: {params}")
# print(f'The model has {count_parameters(model):,} model parameters')
# print(f'The model has {count_parameters(model.features[16]) + count_parameters(model.classifier):,} trainable parameters')
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


train_stats_MobileNetV3 = fit_model(model, 'convnext', train_iterator, valid_iterator, optimizer, loss_criterion,
                                    device,
                                    epochs=100)
# torch.save(model.state_dict(), save_path)
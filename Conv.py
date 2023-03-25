# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Conv
   Description :
   Author :       Silin
   date：          2023/3/17
-------------------------------------------------
   Change Activity:
                   2023/3/17:
-------------------------------------------------
"""
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import random
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
    print('The current GPU device is :', torch.cuda.current_device())

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

SEED = 7


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

t = transforms.Compose([
    transforms.ToTensor(),
    F.normalize,
])
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False,
                                                transform=t)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False,
                                               transform=t)
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))
# feature, label = mnist_test[0]
# print(feature.shape, label)
BATCH_SIZE = 16
# train_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)  # channel, height, weight
test_loader = DataLoader(dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=False)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True)
train_loss = []
train_acc = []
val_loss = []
val_acc = []


class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
        )  # shape of output is batch * 8 * 13 * 13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )  # shape of output is batch * 16 * 5 * 5

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=8 * 5 * 5),
            nn.BatchNorm1d(8 * 5 * 5)
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=8 * 5 * 5, out_features=2 * 5 * 5),
            nn.BatchNorm1d(2 * 5 * 5)
        )

        self.out = nn.Linear(in_features=2 * 5 * 5, out_features=10)

    def forward(self, x):
        x = x  # input layer
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)

        return output


# dropout
class Variant1(nn.Module):
    def __init__(self):
        super(Variant1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )  # shape of output is batch * 8 * 13 * 13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )  # shape of output is batch * 16 * 5 * 5

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=8 * 5 * 5),
            nn.BatchNorm1d(8 * 5 * 5)
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=8 * 5 * 5, out_features=2 * 5 * 5),
            nn.BatchNorm1d(2 * 5 * 5)
        )

        self.out = nn.Linear(in_features=2 * 5 * 5, out_features=10)

    def forward(self, x):
        x = x  # input layer
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)

        return output


# LeakyReLU
class Variant2(nn.Module):
    def __init__(self):
        super(Variant2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
        )  # shape of output is batch * 8 * 13 * 13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )  # shape of output is batch * 16 * 5 * 5

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=8 * 5 * 5),
            nn.BatchNorm1d(8 * 5 * 5)
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=8 * 5 * 5, out_features=2 * 5 * 5),
            nn.BatchNorm1d(2 * 5 * 5)
        )

        self.out = nn.Linear(in_features=2 * 5 * 5, out_features=10)

    def forward(self, x):
        x = x  # input layer
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)

        return output


# Add one convolutional layer
class Variant4(nn.Module):
    def __init__(self):
        super(Variant4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
        )  # shape of output is batch * 8 * 13 * 13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )  # shape of output is batch * 16 * 5 * 5
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # shape of output is batch * 32 * 3 * 3

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=32 * 3 * 3, out_features=16 * 3 * 3),
            nn.BatchNorm1d(16 * 3 * 3)
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=16 * 3 * 3, out_features=4 * 3 * 3),
            nn.BatchNorm1d(4 * 3 * 3)
        )

        self.out = nn.Linear(in_features=4 * 3 * 3, out_features=10)

    def forward(self, x):
        x = x  # input layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)

        return output


def acc(model, dataloader):
    model.eval()
    num_correct = 0
    num_samples = 0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        num_correct += torch.sum(preds == labels)
        num_samples += labels.size(0)
    accuracy = float(num_correct) / float(num_samples)
    return accuracy


def lr_decay(epoch, initial_lr):
    new_lr = initial_lr * (0.5 ** (epoch // 3))
    return new_lr


def train(epochs):
    for fold, (train_ids, validate_ids) in enumerate(kfold.split(mnist_train)):

        model = BaseLine().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-6)   # + baseline: variant 3
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: lr_decay(e, 0.1))

        part_loss = 0
        train_total = []
        val_total = []
        train_accs = []
        val_accs = []
        train_set = Subset(mnist_train, train_ids)
        validate_set = Subset(mnist_train, validate_ids)
        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=validate_set, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(epochs):
            print('-' * 35 + "train" + '-' * 35)
            print(f'Epoch {epoch + 1} of Fold {fold + 1}/{n_splits}')

            train_epoch = 0
            val_epoch = 0
            n = 0
            num = 0
            model.train()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                n += BATCH_SIZE

                optimizer.zero_grad()

                outputs = model(inputs)
                # print("Outputs are ", outputs)
                # print("Labels are ", labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                part_loss += loss.item()
                train_epoch += loss.item()
                if i % 80 == 79:
                    print('[Epoch: %d, Batch: %d] Train loss:%.5f' % (epoch + 1, i + 1, part_loss / 80))
                    part_loss = 0.0

            scheduler.step()

            accuracy = acc(model, train_loader)
            # print(accuracy)
            train_accs.append(accuracy)
            train_total.append(train_epoch / n)

            print('-' * 35 + "validate" + '-' * 35)
            model.eval()
            for i, (inputs, labels) in enumerate(val_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                num += BATCH_SIZE
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_epoch += loss.item()

            val_accs.append(acc(model, val_loader))
            val_total.append(val_epoch / num)

            print('In %d th epoch of %d th Fold, the accuracy of this model on the validate images: %.4f %%' % (
                epoch + 1, fold + 1, acc(model, val_loader) * 100))

        train_loss.append(train_total)
        train_acc.append(train_accs)

        val_loss.append(val_total)
        val_acc.append(val_accs)
        # writer.add_scalar('Loss/train', float(total / n), epoch)

        # print("The learning rate of %d th epoch：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
        # writer.add_scalar('Accuracy/test', float(correct / num), epoch)


def test(model, epochs):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: lr_decay(e, 0.1))

    train_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # print("Outputs are ", outputs)
            # print("Labels are ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), './record/variant3/weightsOfVariant3.pth')

    return acc(model, test_loader)


if __name__ == "__main__":
    # writer = SummaryWriter(log_dir='./logs', comment='UNet+ResNet18')
    # for features, label in train_loader:
    #     features = features.to(device)
    #     print(base(features))
    #     break

    # train(15)
    # path = 'variant4'
    # np.save('./record/' + path + '/train_loss.npy', train_loss)
    # np.save('./record/' + path + '/val_loss.npy', val_loss)
    # np.save('./record/' + path + '/train_acc.npy', train_acc)
    # np.save('./record/' + path + '/val_acc.npy', val_acc)

    best = Variant4().to(device)
    second = Variant2().to(device)
    print(summary(best, (1, 28, 28)))

    # base = BaseLine().to(device)
    # acc_best = test(base, 10)
    # print(acc_best)

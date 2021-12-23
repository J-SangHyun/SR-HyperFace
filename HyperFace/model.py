# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperFace(nn.Module):
    def __init__(self):
        super(HyperFace, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1a = nn.Conv2d(96, 256, kernel_size=(4, 4), stride=(4, 4))
        self.relu1a = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3a = nn.Conv2d(384, 256, kernel_size=(2, 2), stride=(2, 2))
        self.relu3a = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_all = nn.Conv2d(256*3, 192, kernel_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc_full = nn.Linear(192*6*6, 3072)

        self.fc_landmarks1 = nn.Linear(3072, 512)
        self.fc_landmarks2 = nn.Linear(512, 42)

        self.apply(self.weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x1 = self.conv1a(x)
        x1 = self.relu1a(x1)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x2 = self.conv3a(x)
        x2 = self.relu3a(x2)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x3 = self.pool5(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv_all(x)
        x = self.flatten(x)
        x = self.fc_full(x)

        x = self.fc_landmarks1(x)
        x = self.fc_landmarks2(x)

        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    def compute_loss(self, input, target, mask):
        mask = mask.view(-1) == 1.0
        input = input.view(-1)[mask]
        target = target.view(-1)[mask]
        return F.mse_loss(input, target)

import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
import sys

sys.path.append('..')
import random
import cv2

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
#from isplutils import utils
from os import walk

#from IPython.core.display import display, HTML

#display(HTML("<style>.container { width:98% !important; }</style>"))
from sklearn import metrics

import pickle
import argparse
import os
import shutil
import warnings

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

from isplutils import utils, split

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
# from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import ImageChops, Image

from architectures import fornet
from isplutils.data import FrameFaceIterableDataset, load_face
from random import sample

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1,
                                                 output2,
                                                 keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) + (label) *
            torch.pow(torch.clamp(self.margin -
                                  euclidean_distance, min=0.0), 2))

        return loss_contrastive, euclidean_distance


def sample_frames_N(res, N):
    L = min(len(res[0]), len(res[1]))
    indices = sample(range(L), min(L, N))
    out1, out2 = [], []
    for i in indices:
        out1.append(np.asarray(res[0][i].cpu()))
        out2.append(np.asarray(res[1][i].cpu()))
    out1 = torch.Tensor(np.asarray(out1)).to(device)
    out2 = torch.Tensor(np.asarray(out2)).to(device)
    return [out1, out2]


def build_dataset(video_sets, set_label, n_in):
    data_input = np.zeros((0, 2, n_in))
    data_label = np.zeros((0, 1))
    for j, video_set in enumerate(video_sets):
        for i in range(len(video_set)):  # training steps
            out1 = video_set[i][0]
            out2 = video_set[i][1]
            if len(out1) != len(out2):
                continue
            inputs = np.stack(
                (out1, out2),
                axis=1)  # inputs size(num_frame, 2, num_features)
            labels = np.zeros((len(out1), 1)) + set_label[j]
            data_input = np.concatenate((data_input, inputs), axis=0)
            data_label = np.concatenate((data_label, labels), axis=0)
    return TensorDataset(
        torch.from_numpy(data_input).type(torch.float32),
        torch.from_numpy(data_label).type(torch.float32))


def get_Tuned_NN_with_id(device, n_in, n_feature, R):
    model = fornet.IdentityAwareNeuralNetworkNotShared(n_feature,
                                                       n_in - n_feature)
    net_features = model.to(device)
    net_features.load_state_dict(
        torch.load(
            'trained_weight/unsupervised-final-checkpoint-proposed-run-' +
            str(R) + '.pt'))
    return net_features


def Tune_NN_with_id(device, initial_lr, epoch_num, real_videos_train,
                    real_videos_valid, real_videos_test, fake_videos_train,
                    fake_videos_valid, fake_videos_test, n_in, n_feature, R):
    criterion = ContrastiveLoss()
    model = fornet.IdentityAwareNeuralNetworkNotShared(n_feature,
                                                       n_in - n_feature)

    net_features = model.to(device)
    optimizer = optim.Adam(net_features.parameters(), lr=initial_lr)

    train_loss_history, valid_loss_history = [], []
    min_valid_loss = 1e5
    batch_size = 128
    print_mini_batch = 20

    # converting video level dataset to frame level
    train_dataset = build_dataset([real_videos_train, fake_videos_train],
                                  [1, 0], n_in)
    valid_dataset = build_dataset([real_videos_valid, fake_videos_valid],
                                  [1, 0], n_in)
    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)
    validloader = DataLoader(valid_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        last_loss_train = 0.0
        net_features.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            out_feature = net_features(
                [inputs[:, 0, :].to(device), inputs[:, 1, :].to(device)])
            out1_feature, out2_feature = out_feature[0], out_feature[1]
            loss, Dw = criterion(out1_feature, out2_feature, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_mini_batch == print_mini_batch - 1:  # print every 2000 mini-batches
                last_loss_train = running_loss / print_mini_batch
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_mini_batch:.3f}'
                )
                running_loss = 0.0
        train_loss_history.append(last_loss_train)

        running_loss = 0.0
        last_loss_valid = 0.0
        net_features.eval()
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            out_feature = net_features(
                [inputs[:, 0, :].to(device), inputs[:, 1, :].to(device)])
            out1_feature, out2_feature = out_feature[0], out_feature[1]
            loss, Dw = criterion(out1_feature, out2_feature, labels.to(device))

            running_loss += loss.item()
            if i % print_mini_batch == print_mini_batch - 1:  # print every 2000 mini-batches
                last_loss_valid = running_loss / print_mini_batch
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_mini_batch:.3f}'
                )
                running_loss = 0.0
        valid_loss_history.append(last_loss_valid)

        if min_valid_loss >= last_loss_valid:  #np.mean(valid_loss_arr):
            min_valid_loss = last_loss_valid  #np.mean(valid_loss_arr)
            torch.save(
                net_features.state_dict(),
                'trained_weight/unsupervised-final-checkpoint-proposed-run-' +
                str(R) + '.pt')
    net_features.load_state_dict(
        torch.load(
            'trained_weight/unsupervised-final-checkpoint-proposed-run-' +
            str(R) + '.pt'))
    print('training finished')
    return net_features, train_loss_history, valid_loss_history


def get_Tuned_NN_baseline_efficientnet(device, n_in, R):
    net_features = nn.Sequential(nn.BatchNorm1d(num_features=n_in),
                                 nn.Linear(in_features=n_in, out_features=1))
    net_features = net_features.to(device)
    net_features.load_state_dict(
        torch.load('trained_weight/final-baseline-efficientnet-run' + str(R) +
                   '.pt'))
    return net_features


def get_Tuned_NN_baseline_xception(device, n_in, R):
    net_features = nn.Sequential(nn.BatchNorm1d(num_features=n_in),
                                 nn.Linear(in_features=n_in, out_features=1))
    net_features = net_features.to(device)
    net_features.load_state_dict(
        torch.load('trained_weight/final-baseline-xception-run' + str(R) +
                   '.pt'))
    return net_features


def Tune_NN_baseline_efficientnet(device, initial_lr, epoch_num,
                                  real_videos_train, real_videos_valid,
                                  real_videos_test, fake_videos_train,
                                  fake_videos_valid, fake_videos_test, n_in,
                                  R):
    criterion = nn.BCEWithLogitsLoss(
    )  # BCEloss expects (0,1), but logits may surpaass the range
    n_in = n_in
    model = nn.Sequential(nn.BatchNorm1d(num_features=n_in),
                          nn.Linear(in_features=n_in, out_features=1))
    model.load_state_dict(
        torch.load('trained_weight/Efficientnet_classifier_weight.pt'))

    net_features = model.to(device)
    optimizer = optim.Adam(net_features.parameters(), lr=initial_lr)

    train_loss_history, valid_loss_history = [], []
    min_valid_loss = 1e5
    assert len(real_videos_train) == len(fake_videos_train)
    for epoch in range(epoch_num):
        loss_arr = []
        for i in range(len(real_videos_train)):  # training steps
            out1 = real_videos_train[i][0]
            out3 = fake_videos_train[i][0]
            labels = torch.cat(
                (torch.zeros([len(out1), 1]), torch.ones([len(out3), 1])),
                0).to(device)
            out1 = torch.cat((out1, out3), 0)
            out1_feature = net_features(out1.to(device))
            loss = criterion(out1_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_history.append(np.mean(loss_arr))

        valid_loss_arr = []
        for i in range(len(real_videos_valid)):
            out1 = real_videos_valid[i][0]
            out3 = fake_videos_valid[i][0]
            labels = torch.cat(
                (torch.zeros([len(out1), 1]), torch.ones([len(out3), 1])),
                0).to(device)
            out1 = torch.cat((out1, out3), 0)
            out1_feature = net_features(out1.to(device))
            loss = criterion(out1_feature, labels)
            valid_loss_arr.append(loss.item())
        valid_loss_history.append(np.mean(valid_loss_arr))
        if min_valid_loss >= np.mean(valid_loss_arr):
            min_valid_loss = np.mean(valid_loss_arr)
            torch.save(
                net_features.state_dict(),
                'trained_weight/final-baseline-efficientnet-run' + str(R) +
                '.pt')
    net_features.load_state_dict(
        torch.load('trained_weight/final-baseline-efficientnet-run' + str(R) +
                   '.pt'))

    return net_features, train_loss_history, valid_loss_history


def Tune_NN_baseline_xception(device, initial_lr, epoch_num, real_videos_train,
                              real_videos_valid, real_videos_test,
                              fake_videos_train, fake_videos_valid,
                              fake_videos_test, n_in, R):
    criterion = nn.BCEWithLogitsLoss(
    )  # BCEloss expects (0,1), but logits may surpaass the range
    model = nn.Linear(in_features=n_in, out_features=1)
    model.load_state_dict(
        torch.load('trained_weight/Xception_classifier_weight.pt'))

    net_features = model.to(device)
    optimizer = optim.Adam(net_features.parameters(), lr=initial_lr)

    train_loss_history, valid_loss_history = [], []
    min_valid_loss = 1e5
    assert len(real_videos_train) == len(fake_videos_train)
    for epoch in range(epoch_num):
        loss_arr = []
        for i in range(len(real_videos_train)):  # training steps
            out1 = real_videos_train[i][0]
            out3 = fake_videos_train[i][0]
            labels = torch.cat(
                (torch.zeros([len(out1), 1]), torch.ones([len(out3), 1])),
                0).to(device)
            out1 = torch.cat((out1, out3), 0)
            out1_feature = net_features(out1.to(device))
            loss = criterion(out1_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_history.append(np.mean(loss_arr))

        valid_loss_arr = []
        for i in range(len(real_videos_valid)):
            out1 = real_videos_valid[i][0]
            out3 = fake_videos_valid[i][0]
            labels = torch.cat(
                (torch.zeros([len(out1), 1]), torch.ones([len(out3), 1])),
                0).to(device)
            out1 = torch.cat((out1, out3), 0)
            out1_feature = net_features(out1.to(device))
            loss = criterion(out1_feature, labels)
            valid_loss_arr.append(loss.item())
        valid_loss_history.append(np.mean(valid_loss_arr))
        if min_valid_loss >= np.mean(valid_loss_arr):
            min_valid_loss = np.mean(valid_loss_arr)
            torch.save(
                net_features.state_dict(),
                'trained_weight/final-baseline-xception-run' + str(R) + '.pt')
    net_features.load_state_dict(
        torch.load('trained_weight/final-baseline-xception-run' + str(R) +
                   '.pt'))

    return net_features, train_loss_history, valid_loss_history

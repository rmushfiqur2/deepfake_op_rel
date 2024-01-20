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
from architectures import fornet,weights
from isplutils import utils
from os import walk

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageChops, Image

from architectures import fornet
from isplutils.data import FrameFaceIterableDataset, load_face
from random import sample

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


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

def Tune_NN_with_id(device, initial_lr, epoch_num, real_videos_train, real_videos_valid, real_videos_test, fake_videos_train,
            fake_videos_valid, fake_videos_test, n_in=1792):
    eer_arr_old, eer_arr_new, eer_arr_frame_old, eer_arr_frame_new = [], [], [], []
    criterion = ContrastiveLoss()
    t = 0.8  # thresholddd
    n_in = 1792
    n_out = 50
    model = nn.Sequential(nn.Linear(n_in, n_out))

    # net_features = NN_for_features().to(device)
    net_features = model.to(device)
    optimizer = optim.Adam(net_features.parameters(), lr=initial_lr)

    train_loss_history, valid_loss_history = [], []
    min_valid_loss = 1e5
    for epoch in range(epoch_num):
        loss_arr = []
        for i in range(len(real_videos_train)):  # training steps
            out1 = real_videos_train[i][0]
            out2 = real_videos_train[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.ones([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for i in range(len(fake_videos_train)):
            out1 = fake_videos_train[i][0]
            out2 = fake_videos_train[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.zeros([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_history.append(np.mean(loss_arr))

        # if epoch%100 == 0:
        # print(epoch)

        valid_loss_arr = []
        for i in range(len(real_videos_valid)):
            out1 = real_videos_valid[i][0]
            out2 = real_videos_valid[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.ones([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            valid_loss_arr.append(loss.item())
        for i in range(len(fake_videos_valid)):
            out1 = fake_videos_valid[i][0]
            out2 = fake_videos_valid[i][1]
            if len(out1) != len(out2):
                continue
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            labels = torch.zeros([len(out1), 1]).to(device)
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            valid_loss_arr.append(loss.item())
        valid_loss_history.append(np.mean(valid_loss_arr))
        if min_valid_loss >= np.mean(valid_loss_arr):
            min_valid_loss = np.mean(valid_loss_arr)
            # net_tuned = net_features
            torch.save(net_features.state_dict(), 'temp-checkpoint.pt')
    net_features.load_state_dict(torch.load('temp-checkpoint.pt'))

    return net_features, train_loss_history, valid_loss_history
def Tune_NN(device, initial_lr, epoch_num, real_videos_train, real_videos_valid, real_videos_test, fake_videos_train, fake_videos_valid, fake_videos_test, n_in=1792):

    eer_arr_old, eer_arr_new, eer_arr_frame_old, eer_arr_frame_new = [], [], [], []
    criterion = ContrastiveLoss()
    t = 0.8 #thresholddd
    n_in = n_in
    n_out = 50
    model = nn.Sequential(nn.Linear(n_in, n_out))

    #net_features = NN_for_features().to(device)
    net_features = model.to(device)
    optimizer = optim.Adam(net_features.parameters(), lr=initial_lr)
    
    train_loss_history, valid_loss_history = [], []
    min_valid_loss = 1e5
    for epoch in range(epoch_num):
        loss_arr = []
        for i in range(len(real_videos_train)): # training steps
            out1 = real_videos_train[i][0]
            out2 = real_videos_train[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.ones([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for i in range(len(fake_videos_train)):
            out1 = fake_videos_train[i][0]
            out2 = fake_videos_train[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.zeros([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            loss_arr.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_history.append(np.mean(loss_arr))
        
        #if epoch%100 == 0:
            #print(epoch)

        valid_loss_arr = []
        for i in range(len(real_videos_valid)):
            out1 = real_videos_valid[i][0]
            out2 = real_videos_valid[i][1]
            if len(out1) != len(out2):
                continue
            labels = torch.ones([len(out1), 1]).to(device)
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            valid_loss_arr.append(loss.item())
        for i in range(len(fake_videos_valid)):
            out1 = fake_videos_valid[i][0]
            out2 = fake_videos_valid[i][1]
            if len(out1) != len(out2):
                continue
            out1_feature = net_features(out1.to(device))
            out2_feature = net_features(out2.to(device))
            labels = torch.zeros([len(out1), 1]).to(device)
            loss, Dw = criterion(out1_feature, out2_feature, labels)
            valid_loss_arr.append(loss.item())
        valid_loss_history.append(np.mean(valid_loss_arr))
        if min_valid_loss >= np.mean(valid_loss_arr):
            min_valid_loss = np.mean(valid_loss_arr)
            #net_tuned = net_features
            torch.save(net_features.state_dict(), 'temp-checkpoint.pt')
    net_features.load_state_dict(torch.load('temp-checkpoint.pt'))

    return net_features, train_loss_history, valid_loss_history
import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import get_network
from models.common import post_process_output
from models.salgrasp import MultiTaskLoss
from tensorboard.plugins import projector
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.dataset_processing.grasp import visualise_output
from utils.metric import AvgMeter, cal_maxf, cal_pr_mae_meanf
from utils.visualisation.confusion import (count_elements, histogram_plot,
										   plot_confusion_matrix)
from utils.visualisation.gridshow import gridshow

Dataset = get_dataset('jacquard_depth')

f = json.load(open('k_split_indices.txt', 'rb'))
train_indices = f[str('0')]['train']
test_indices = f[str('0')]['test']

input_channels = 3
transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5)]))

mtgcnn = get_network('sg2')

net = mtgcnn(input_channels=input_channels)
device = torch.device("cuda:0")
net = net.to(device)

train_dataset = Dataset('/media/will/research/Jacquard/data',
							random_rotate=True, random_zoom=True, include_depth=0,
							include_rgb=1, shuffle=True, 
							transform=transformations, ids=train_indices)

train_data = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=1,
	shuffle=True,
	num_workers=1
)

net.train()

for x, y, targets, _, _, _ in train_data:
	xc = x.to(device)
	yc = [yy.to(device) for yy in y]
	target = targets.to(device)
	import ipdb; ipdb.set_trace()
	lossd = net.compute_loss(xc, target, yc)

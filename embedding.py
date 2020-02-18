import argparse
import datetime
import os
import sys
import numpy as np
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import get_network
from utils.data import get_dataset
from utils.dataset_processing import evaluation


def parse_args():
	parser = argparse.ArgumentParser(description='Model embeddings')

	# Network
	parser.add_argument('--network', type=str, default='mtgcnn2', help='Network Name in .models')

	# Dataset & Data & Training
	parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
	parser.add_argument('--dataset-path', type=str, help='Path to dataset')
	parser.add_argument('--network-path', type=str, help='Path to dataset')
	parser.add_argument('--json', type=str, help='Path to image classifications', default='annotations/coco.json')
	parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
	parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
	parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
	parser.add_argument('--random_seed', type=int, default=42, help='random seed for splitting the dataset into train and test sets')
	parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before splitting')
	parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

	# Logging etc.
	parser.add_argument('--description', type=str, default='embedding', help='embedding description')
	parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')

	args = parser.parse_args()
	return args

def load_images(files):
	images = []
	for filename in files:
		img  = cv2.imread(filename)
		if img is not None:
			images.append(img[:300,:300,:])
	return images

def run():

	args = parse_args()

	dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
	net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

	writer = SummaryWriter(os.path.join(args.logdir, net_desc))

	Dataset = get_dataset(args.dataset)

	val_dataset = val_dataset = Dataset(args.dataset_path, json=args.json, split=args.split,
							random_rotate=False, random_zoom=False, include_depth=args.use_depth,
							include_rgb=args.use_rgb, train=True, shuffle=args.shuffle, seed=args.random_seed)
	val_data = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
	classes = val_dataset.nms

	input_channels = 1*args.use_depth + 3*args.use_rgb
	device = torch.device("cuda:0")
	model = get_network(args.network)

	net = model().to(device)
	net = torch.load(args.network_path)
	net.eval()

	exampleimages, examplelabels, _, _, _= next(iter(val_data))
	exampleclasses = [classes[lab.item()] for lab in examplelabels[-1]]
	grid = torchvision.utils.make_grid(exampleimages, normalize=True)
	writer.add_image('valexampleimages', grid, global_step=0)
	print('validation example classes', exampleclasses)

	images = None
	targets = []

	for x, y, _, _, _ in val_data:
		if images is not None:
			images = torch.cat((images, x), 0)
		else:
			images = x
		targets.append(classes[y[-1].item()])

	if images.data.shape[1] == 4:
		writer.add_embedding(images.view(-1, input_channels*300*300), metadata=targets, label_img=images.data[:,1:,:,:], global_step=0)
	else:
		writer.add_embedding(images.view(-1, input_channels*300*300), metadata=targets, label_img=images.data, global_step=0)

	print('initial embedding completed')

	class_out = None

	for i in images:
		i = i.unsqueeze(0).to(device)
		if class_out is not None:
			class_out = torch.cat((class_out, net(i)[-1][0].unsqueeze(0)), 0)
		else:
			class_out = net(i)[-1][0].unsqueeze(0)

	if images.data.shape[1] == 4:
		writer.add_embedding(class_out, metadata=targets, label_img=images.data[:,1:,:,:], global_step=1)
	else:
		writer.add_embedding(class_out, metadata=targets, label_img=images.data, global_step=1)

	writer.flush()

	writer.close()

if __name__ == '__main__':
	run()

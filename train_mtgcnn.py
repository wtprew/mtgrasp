import argparse
import datetime
import logging
import os
import sys

import cv2
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
from tensorboard.plugins import projector
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import get_network
from models.common import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.confusion import plot_confusion_matrix
from utils.visualisation.gridshow import gridshow

logging.basicConfig(level=logging.INFO)

# cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

def parse_args():
	parser = argparse.ArgumentParser(description='Train MTGCNN')

	# Network
	parser.add_argument('--network', type=str, default='mtgcnn', help='Network Name in .models')

	# Dataset & Data & Training
	parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
	parser.add_argument('--dataset-path', type=str, help='Path to dataset')
	parser.add_argument('--json', type=str, help='Path to image classifications', default='annotations/coco.json')
	parser.add_argument('--loss_type', type=str, default='grasp', help='Type of loss function to use ("grasp", "class", "combined")')
	parser.add_argument('--grasp_weight', type=float, default=1.0, help='Loss weight to modify the grasp weight')
	parser.add_argument('--class_weight', type=float, default=1.0, help='Loss weight to modify the class weight')
	parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
	parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
	parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
	parser.add_argument('--random_seed', type=int, default=42, help='random seed for splitting the dataset into train and test sets')
	parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before splitting')
	parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

	parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
	parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
	parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
	parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

	# Logging etc.
	parser.add_argument('--description', type=str, default='', help='Training description')
	parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
	parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
	parser.add_argument('--vis', action='store_true', help='Visualise the training process')

	args = parser.parse_args()
	return args


def validate(net, loss_type, device, val_data, batches_per_epoch, grasp_weighting=1.0, class_weighting=1.0):
	"""
	Run validation.
	:param net: Network
	:param device: Torch device
	:param val_data: Validation Dataset
	:param batches_per_epoch: Number of batches to run
	:return: Successes, Failures and Losses
	"""
	net.eval()

	results = {
		'graspcorrect': 0,
		'graspfailed': 0,
		'classcorrect': 0,
		'classfailed': 0,
		'grasploss': 0,
		'classloss': 0,
		'losses': {
		},
		'pred': [],
		'label': [],
	}

	ld = len(val_data)
	predicted = []
	labels = []

	with torch.no_grad():
		batch_idx = 0
		while batch_idx < batches_per_epoch:
			for x, y, didx, rot, zoom_factor in val_data:
				batch_idx += 1
				if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
					break

				xc = x.to(device)
				yc = [yy.to(device) for yy in y]
				lossd = net.compute_loss(xc, yc, grasp_weight=grasp_weighting, class_weight=class_weighting)

				grasploss = lossd['loss']['grasp']
				classloss = lossd['loss']['class']

				results['grasploss'] += grasploss.item()/ld
				results['classloss'] += classloss.item()/ld
				for ln, l in lossd['losses'].items():
					if ln not in results['losses']:
						results['losses'][ln] = 0
					results['losses'][ln] += l.item()/ld

				q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
															lossd['pred']['sin'], lossd['pred']['width'])

				s = evaluation.calculate_iou_match(q_out, ang_out,
												   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
												   no_grasps=1,
												   grasp_width=w_out,
												   )

				if s:
					results['graspcorrect'] += 1
				else:
					results['graspfailed'] += 1
				
				#test classification
				_, class_pred = torch.max(lossd['pred']['class'], 1)
				pred = class_pred.item()
				predicted.append(pred)
				label = y[-1].item()
				labels.append(label)
				if pred == label:
					results['classcorrect'] += 1
				else:
					results['classfailed'] += 1
	
	results['pred'] = predicted
	results['label'] = labels
	
	return results


def train(epoch, loss_type, net, device, train_data, optimizer, batches_per_epoch, grasp_weighting=1.0, class_weighting=1.0, vis=False):
	"""
	Run one training epoch
	:param epoch: Current epoch
	:param net: Network
	:param device: Torch device
	:param train_data: Training Dataset
	:param optimizer: Optimizer
	:param batches_per_epoch:  Data batches to train on
	:param vis:  Visualise training progress
	:return:  Average Losses for Epoch
	"""
	results = {
		'grasploss': 0,
		'classloss': 0,
		'losses': {
		}
	}

	net.train()

	batch_idx = 0
	# Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
	while batch_idx < batches_per_epoch:
		for x, y, _, _, _ in train_data:
			batch_idx += 1
			if batch_idx >= batches_per_epoch:
				break

			xc = x.to(device)
			yc = [yy.to(device) for yy in y]
			lossd = net.compute_loss(xc, yc, grasp_weight=grasp_weighting, class_weight=class_weighting)

			grasploss = lossd['loss']['grasp']
			classloss = lossd['loss']['class']

			if batch_idx % 100 == 0:
				logging.info('Epoch: {}, Batch: {}, Grasp Loss: {:0.4f}, Class Loss {:0.4f}'.format(epoch, batch_idx, grasploss.item(), classloss.item()))

			results['grasploss'] += grasploss.item()
			results['classloss'] += classloss.item()
			for ln, l in lossd['losses'].items():
				if ln not in results['losses']:
					results['losses'][ln] = 0
				results['losses'][ln] += l.item()

			optimizer.zero_grad()
			if loss_type == 'grasp':
				grasploss.backward()
			elif loss_type == 'class':
				classloss.backward()
			elif loss_type == 'combined':
				grasploss.backward(retain_graph=True)
				classloss.backward()
			else:
				raise TypeError('--loss_type must be either "grasp", "class", or "combined"')
			optimizer.step()

			# Display the images and add network to tensorboard graph
			# if vis:
				# imgs = []
				# n_img = min(4, x.shape[0])
				# for idx in range(n_img):
				# 	imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
				# 		x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
				# gridshow('Display', imgs,
				# 		 [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
				# 		 [cv2.COLORMAP_BONE] * 10 * n_img, 10)
				# cv2.waitKey(1)

	results['grasploss'] /= batch_idx
	results['classloss'] /= batch_idx
	for l in results['losses']:
		results['losses'][l] /= batch_idx

	return results


def run():
	args = parse_args()

	# Set-up output directories
	dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
	net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

	save_folder = os.path.join(args.outdir, net_desc)
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	writer = SummaryWriter(os.path.join(args.logdir, net_desc))

	# Load Dataset
	logging.info('Loading {} Dataset...'.format(args.dataset.title()))
	Dataset = get_dataset(args.dataset)

	print('Training dataset loading')
	train_dataset = Dataset(args.dataset_path, json=args.json, split=args.split,
						random_rotate=True, random_zoom=True, include_depth=args.use_depth,
						include_rgb=args.use_rgb, train=True, shuffle=args.shuffle, seed=args.random_seed)
	classes = train_dataset.nms
	supercategories = train_dataset.supcats
	print('target classes', classes, 'target_superclasses', supercategories)
	print('Validation set loading')
	val_dataset = Dataset(args.dataset_path, json=args.json, split=args.split,
						random_rotate=True, random_zoom=True, include_depth=args.use_depth,
						include_rgb=args.use_rgb, train=False, shuffle=args.shuffle, seed=args.random_seed)

	train_data = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers
	)

	val_data = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=args.num_workers
	)
	logging.info('Done')

	# Load the network
	logging.info('Loading Network...')
	input_channels = 1*args.use_depth + 3*args.use_rgb
	mtgcnn = get_network(args.network)

	net = mtgcnn(input_channels=input_channels, num_classes=len(train_dataset.cats))
	device = torch.device("cuda:0")
	net = net.to(device)
	optimizer = optim.Adam(net.parameters())
	logging.info('Done')

	# display a set of example images
	exampleimages, examplelabels, _, _, _ = next(iter(train_data))
	exampleclasses = [classes[lab] for lab in examplelabels[-1]]
	grid = torchvision.utils.make_grid(exampleimages, normalize=True)
	writer.add_image('trainexampleimages', grid)
	print('training example classes', exampleclasses)

	exampleimages, examplelabels, _, _, _ = next(iter(val_data))
	exampleclasses = [classes[lab] for lab in examplelabels[-1]]
	grid = torchvision.utils.make_grid(exampleimages, normalize=True)
	writer.add_image('valexampleimages', grid)
	print('validation example classes', exampleclasses)

	# Print model architecture.
	writer.add_graph(net, exampleimages.to(device))
	
	summary(net, (input_channels, 300, 300))
	f = open(os.path.join(save_folder, 'arch.txt'), 'w')
	sys.stdout = f
	summary(net, (input_channels, 300, 300))
	sys.stdout = sys.__stdout__
	f.close()

	best_iou = 0.0
	best_classification = 0.0
	for epoch in range(args.epochs):
		logging.info('Beginning Epoch {:02d}'.format(epoch))
		train_results = train(epoch, args.loss_type, net, device, train_data, optimizer, args.batches_per_epoch, grasp_weighting=args.grasp_weight, class_weighting=args.class_weight, vis=args.vis)

		# Log training losses to tensorboard
		writer.add_scalar('loss/grasp_loss', train_results['grasploss'], epoch)
		writer.add_scalar('loss/class_loss', train_results['classloss'], epoch)

		for n, l in train_results['losses'].items():
			writer.add_scalar('train_loss/' + n, l, epoch)

		# Run Validation
		logging.info('Validating...')
		test_results = validate(net, args.loss_type, device, val_data, args.val_batches, grasp_weighting=args.grasp_weight, class_weighting=args.class_weight)
		logging.info('IoU results %d/%d = %f' % (test_results['graspcorrect'], test_results['graspcorrect'] + test_results['graspfailed'],
									test_results['graspcorrect']/(test_results['graspcorrect']+test_results['graspfailed'])))
		logging.info('Classification results %d/%d = %f' % (test_results['classcorrect'], test_results['classcorrect'] + test_results['classfailed'],
									test_results['classcorrect']/(test_results['classcorrect']+test_results['classfailed'])))

		# Log validation results to tensorbaord
		writer.add_scalar('loss/IOU', test_results['graspcorrect'] / (test_results['graspcorrect'] + test_results['graspfailed']), epoch)
		writer.add_scalar('loss/class_accuracy', test_results['classcorrect'] / (test_results['classcorrect'] + test_results['classfailed']), epoch)
		writer.add_scalar('val_loss/val_class_loss', test_results['classloss'], epoch)
		writer.add_scalar('val_loss/val_grasp_loss', test_results['grasploss'], epoch)
		for n, l in test_results['losses'].items():
			writer.add_scalar('val_loss/' + n, l, epoch)

		figure = plot_confusion_matrix(test_results['label'], test_results['pred'], classes)
		writer.add_figure('Confusion_matrix', figure, global_step=epoch)

		# Save best performing network
		iou = test_results['graspcorrect'] / (test_results['graspcorrect'] + test_results['graspfailed'])
		classification = test_results['classcorrect'] / (test_results['classcorrect'] + test_results['classfailed'])
		if iou > best_iou or classification > best_classification or epoch == 0 or (epoch % 10) == 0:
			torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_class_%0.2f' % (epoch, iou, classification)))
			best_iou = iou
			best_classification = classification

		writer.flush()
	writer.close()


if __name__ == '__main__':
	run()

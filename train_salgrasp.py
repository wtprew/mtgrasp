import argparse
import datetime
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
from tensorboard.plugins import projector
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import get_network
from models.common import post_process_output
from models.salgrasp import MultiTaskLoss
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.dataset_processing.grasp import visualise_output
from utils.visualisation.confusion import plot_confusion_matrix, count_elements, histogram_plot
from utils.visualisation.gridshow import gridshow
from utils.metric import cal_pr_mae_meanf, cal_maxf, AvgMeter

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
	parser.add_argument('--superclass', action='store_true', help='use superclasses for training')
	parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
	parser.add_argument('--random-seed', type=int, default=42, help='random seed for splitting the dataset into train and test sets')
	parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before splitting')
	parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

	parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
	parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
	parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
	parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

	# Logging etc.
	parser.add_argument('--description', type=str, default='', help='Training description')
	parser.add_argument('--outdir', type=str, default='output/models/Saliency_mt', help='Training Output Directory')
	parser.add_argument('--logdir', type=str, default='tensorboard/Saliency_mt', help='Log directory')
	parser.add_argument('--vis', action='store_true', help='Visualise the training process')

	args = parser.parse_args()
	return args


def validate(net, loss_type, device, val_data, mt, batches_per_epoch, grasp_weighting=1.0, class_weighting=1.0, title=None):
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
		'loss': 0,
		'logvars': [],
		'losses': {
		},
		'maxf': 0,
		'meanf': 0,
		'mae': 0,
	}

	ld = len(val_data)
	pres = [AvgMeter() for _ in range(256)]
	recs = [AvgMeter() for _ in range(256)]
	meanfs = AvgMeter()
	maes = AvgMeter()
	to_pil = transforms.ToPILImage()

	with torch.no_grad():
		batch_idx = 0
		while batch_idx < batches_per_epoch:
			for x, targets, y, didx, rot, zoom_factor in val_data:
				batch_idx += 1
				if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
					break

				xc = x.to(device)
				target = targets.to(device)
				yc = [yy.to(device) for yy in y]
				lossd = net.compute_loss(xc, target, yc, grasp_weight=grasp_weighting, class_weight=class_weighting)

				if loss_type == 'kendall':
					loss, log_vars = mt(lossd['loss']['grasp'], lossd['loss']['class'])
					# loss, log_vars = mt(lossd['losses']['p_loss'], lossd['losses']['cos_loss'], lossd['losses']['sin_loss'], lossd['losses']['width_loss'], lossd['losses']['class_loss']) #all output loss
					results['loss'] += loss.item()/ld
					results['logvars'] = [n/ld for n in log_vars]

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
				
				if batch_idx == 1:
					fig = visualise_output(xc[0], target[0], net, grasp_success=s, title=str(title))

				predsal = np.asarray(to_pil(lossd['pred']['class'].cpu().detach().squeeze()))
				gt = np.asarray(to_pil(target.cpu().detach().squeeze()))

				ps, rs, mae, meanf = cal_pr_mae_meanf(predsal, gt)
				for pidx, pdata in enumerate(zip(ps, rs)):
					p, r = pdata
					pres[pidx].update(p)
					recs[pidx].update(r)
				maes.update(mae)
				meanfs.update(meanf)

	maxf = cal_maxf([pre.avg for pre in pres], [rec.avg for rec in recs])
	results['maxf'] = maxf
	results['meanf'] = meanfs.avg
	results['mae'] = maes.avg

	return results, fig


def train(epoch, loss_type, net, device, train_data, mt, optimizer, batches_per_epoch, grasp_weighting=1.0, class_weighting=1.0, vis=False):
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
		'loss': 0,
		'grasploss': 0,
		'classloss': 0,
		'losses': {
		}
	}

	net.train()

	batch_idx = 0
	# Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
	while batch_idx < batches_per_epoch:
		for x, targets, y, _, _, _ in train_data:
			batch_idx += 1
			if batch_idx >= batches_per_epoch:
				break

			xc = x.to(device)
			target = targets.to(device)
			yc = [yy.to(device) for yy in y]
			lossd = net.compute_loss(xc, target, yc, grasp_weight=grasp_weighting, class_weight=class_weighting)

			if loss_type == 'kendall':
				loss, log_vars = mt(lossd['loss']['grasp'], lossd['loss']['class']) #two task loss
				# loss, log_vars = mt(lossd['losses']['p_loss'], lossd['losses']['cos_loss'], lossd['losses']['sin_loss'], lossd['losses']['width_loss'], lossd['losses']['class_loss']) #all output loss

			grasploss = lossd['loss']['grasp']
			classloss = lossd['loss']['class']

			if batch_idx % 100 == 0:
				if loss_type == 'kendall':
					print('Epoch: {}, Batch: {}, Kendall Loss {:0.4f}, Grasp Loss: {:0.4f}, Class Loss {:0.4f}'.format(epoch, batch_idx, loss.item(), grasploss.item(), classloss.item()))
				else:
					print('Epoch: {}, Batch: {}, Grasp Loss: {:0.4f}, Class Loss {:0.4f}'.format(epoch, batch_idx, grasploss.item(), classloss.item()))

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
			elif loss_type =='kendall':
				results['loss'] += loss.item()
				loss.backward()
			else:
				raise TypeError('--loss_type must be either "grasp", "class", or "combined"')
			optimizer.step()

	results['loss'] /= batch_idx
	results['grasploss'] /= batch_idx
	results['classloss'] /= batch_idx
	for l in results['losses']:
		results['losses'][l] /= batch_idx

	return results


def run():
	args = parse_args()

	print("args called: ", args)

	# Set-up output directories
	dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
	net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

	save_folder = os.path.join(args.outdir, net_desc)
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	writer = SummaryWriter(os.path.join(args.logdir, net_desc))

	# Load Dataset
	print('Loading {} Dataset...'.format(args.dataset.title()))
	Dataset = get_dataset(args.dataset)

	input_channels = 1*args.use_depth + 3*args.use_rgb
	transformations = transforms.Compose([transforms.ToTensor()])
	
	if args.dataset == 'jacquard_sal':
		print('Training dataset loading')
		train_dataset = Dataset(args.dataset_path, split=args.split,
							random_rotate=True, random_zoom=True, include_depth=args.use_depth,
							include_rgb=args.use_rgb, train=True, shuffle=args.shuffle, 
							transform=transformations, seed=args.random_seed)
		print('Validation set loading')
		val_dataset = Dataset(args.dataset_path, split=args.split,
							random_rotate=True, random_zoom=False, include_depth=args.use_depth,
							include_rgb=args.use_rgb, train=False, shuffle=args.shuffle,
							transform=transformations, seed=args.random_seed)
	else:
		print('Training dataset loading')
		train_dataset = Dataset(args.dataset_path, json=args.json, split=args.split,
							random_rotate=True, random_zoom=True, include_depth=args.use_depth,
							include_rgb=args.use_rgb, train=True, shuffle=args.shuffle, 
							transform=transformations, seed=args.random_seed)
		print('Validation set loading')
		val_dataset = Dataset(args.dataset_path, json=args.json, split=args.split,
							random_rotate=True, random_zoom=False, include_depth=args.use_depth,
							include_rgb=args.use_rgb, train=False, shuffle=args.shuffle,
							transform=transformations, seed=args.random_seed)
	
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
	print('Done')

	# Load the network
	print('Loading Network...')
	mtgcnn = get_network(args.network)

	net = mtgcnn(input_channels=input_channels)
	device = torch.device("cuda:0")
	net = net.to(device)

	if args.loss_type == 'kendall':
		multitask = MultiTaskLoss(2).to(device)
		parameters = list(net.parameters()) + list(multitask.parameters())
	else:
		multitask = None
		parameters = list(net.parameters())
	optimizer = optim.Adam(parameters)
	print('Done')

	# display a set of example images
	exampleimages, examplelabels, _, _, _, _ = next(iter(train_data))
	grid = torchvision.utils.make_grid(exampleimages, normalize=True)
	writer.add_image('trainexampleimages', grid)
	grid = torchvision.utils.make_grid(examplelabels, normalize=True)
	writer.add_image('trainexamplesaliency', grid)

	exampleimages, examplelabels, _, _, _, _ = next(iter(val_data))
	grid = torchvision.utils.make_grid(exampleimages, normalize=True)
	writer.add_image('valexampleimages', grid)
	grid = torchvision.utils.make_grid(examplelabels, normalize=True)
	writer.add_image('valexamplesaliency', grid)

	# Print model architecture.
	writer.add_graph(net, exampleimages.to(device))
	
	summary(net, (input_channels, 300, 300))
	f = open(os.path.join(save_folder, 'arch.txt'), 'w')
	sys.stdout = f
	summary(net, (input_channels, 300, 300))
	sys.stdout = sys.__stdout__
	f.close()

	best_iou = {'epoch': 0, 'iou': 0.0, 'maxf': 0.0, 'meanf': 0.0, 'mae': 0.0,}
	best_maxf = {'epoch': 0, 'iou': 0.0, 'maxf': 0.0, 'meanf': 0.0, 'mae': 0.0,}
	best_meanf = {'epoch': 0, 'iou': 0.0, 'maxf': 0.0, 'meanf': 0.0, 'mae': 0.0,}
	best_mae = {'epoch': 0, 'iou': 0.0, 'maxf': 0.0, 'meanf': 0.0, 'mae': 0.0,}
	for epoch in range(args.epochs):
		print('Beginning Epoch {:02d}'.format(epoch))
		train_results = train(epoch, args.loss_type, net, device, train_data, multitask, optimizer, args.batches_per_epoch, grasp_weighting=args.grasp_weight, class_weighting=args.class_weight, vis=args.vis)

		# Log training losses to tensorboard
		writer.add_scalar('loss/grasp_loss', train_results['grasploss'], epoch)
		writer.add_scalar('loss/class_loss', train_results['classloss'], epoch)

		for n, l in train_results['losses'].items():
			writer.add_scalar('train_loss/' + n, l, epoch)

		# Run Validation
		print('Validating...')
		test_results, fig = validate(net, args.loss_type, device, val_data, multitask, args.val_batches, grasp_weighting=args.grasp_weight, class_weighting=args.class_weight, title=args.description)
		print('IoU results %d/%d = %f' % (test_results['graspcorrect'], test_results['graspcorrect'] + test_results['graspfailed'],
									test_results['graspcorrect']/(test_results['graspcorrect']+test_results['graspfailed'])))
		print('Log_vars: ',  test_results['logvars'])
		
		maxf = test_results['maxf']
		meanf = test_results['meanf']
		mae = test_results['mae']
		
		print('Maxf: ', maxf, ' Meanf: ', meanf, ' MAE: ', mae)

		# Log validation results to tensorbaord
		writer.add_scalar('loss/IOU', test_results['graspcorrect'] / (test_results['graspcorrect'] + test_results['graspfailed']), epoch)
		writer.add_scalar('val_loss/val_class_loss', test_results['classloss'], epoch)
		writer.add_scalar('val_loss/val_grasp_loss', test_results['grasploss'], epoch)
		writer.add_scalar('val_loss/val_maxf', maxf, epoch)
		writer.add_scalar('val_loss/val_meanf',meanf, epoch)
		writer.add_scalar('val_loss/val_mae', mae, epoch)

		for n, l in test_results['losses'].items():
			writer.add_scalar('val_loss/' + n, l, epoch)

		# Save best performing network
		iou = test_results['graspcorrect'] / (test_results['graspcorrect'] + test_results['graspfailed'])
		# if iou > best_iou or maxf > best_maxf or epoch == 0 or (epoch % 10) == 0:
		torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_maxf_%0.2f_meanf_%0.2f_mae_%0.2f' % (epoch, iou, maxf, meanf, mae)))
		if iou > best_iou['iou']:		
			best_iou = {'epoch': epoch, 'iou': iou, 'maxf': maxf, 'meanf': meanf, 'mae': mae}
		if maxf > best_maxf['maxf']:
			best_maxf = {'epoch': epoch, 'iou': iou, 'maxf': maxf, 'meanf': meanf, 'mae': mae}
		if meanf > best_meanf['meanf']:
			best_meanf = {'epoch': epoch, 'iou': iou, 'maxf': maxf, 'meanf': meanf, 'mae': mae}
		if mae > best_mae['mae']:
			best_mae = {'epoch': epoch, 'iou': iou, 'maxf': maxf, 'meanf': meanf, 'mae': mae}
		writer.flush()

		fig.savefig(os.path.join(save_folder, str(epoch)+'_'+str(iou)))
		plt.close(fig)

	for i in best_iou:
		print('Best IOU score')
		print(i, best_iou[i])
	for i in best_maxf:
		print('Best MaxF score')
		print(i, best_maxf[i])
	for i in best_meanf:
		print('Best MeanF score')
		print(i, best_meanf[i])
	for i in best_mae:
		print('Best MAE score')
		print(i, best_mae[i])
	writer.close()


if __name__ == '__main__':
	run()

import argparse
import logging

import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.common import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.dataset_processing.grasp import visualise_output
from utils.metric import AvgMeter, cal_maxf, cal_pr_mae_meanf

logging.basicConfig(level=logging.INFO)

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate SG-CNN')

	# Network
	parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

	# Dataset & Data & Training
	parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
	parser.add_argument('--dataset-path', type=str, help='Path to dataset')
	parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
	parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
	parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
	parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
	parser.add_argument('--random-seed', type=int, default=42, help='random seed for splitting the dataset into train and test sets')
	parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before splitting')
	parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

	parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
	parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
	parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
	parser.add_argument('--vis', action='store_true', help='Visualise the network output')

	args = parser.parse_args()

	if args.jacquard_output and args.dataset != 'jacquard_sal':
		raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
	if args.jacquard_output and args.augment:
		raise ValueError('--jacquard-output can not be used with data augmentation.')

	return args


if __name__ == '__main__':
	args = parse_args()

	# Load Network
	net = torch.load(args.network)
	device = torch.device("cuda:0")

	# Load Dataset
	print('Loading {} Dataset...'.format(args.dataset.title()))
	Dataset = get_dataset(args.dataset)
	transformations = transforms.Compose([transforms.ToTensor()])

	if args.dataset == 'jacquard_skfold':
		test_dataset = Dataset(args.dataset_path,
							random_rotate=True, random_zoom=False, include_depth=args.use_depth,
							include_rgb=args.use_rgb, shuffle=args.shuffle,
							transform=transformations)
	else:
		test_dataset = Dataset(args.dataset_path, json=args.json,
							random_rotate=True, random_zoom=False, include_depth=args.use_depth,
							include_rgb=args.use_rgb, shuffle=args.shuffle,
							transform=transformations)

	test_data = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=args.num_workers)
	
	print('Done')

	results = {'grasp':{'correct': 0, 'failed': 0}, 'maxf': 0, 'meanf': 0, 'mae': 0}
	ld = len(test_data)
	pres = [AvgMeter() for _ in range(256)]
	recs = [AvgMeter() for _ in range(256)]
	meanfs = AvgMeter()
	maes = AvgMeter()

	if args.jacquard_output:
		jo_fn = args.network + '_jacquard_output.txt'
		with open(jo_fn, 'w') as f:
			pass

	with torch.no_grad():
		for idx, (x, targets, y, didx, rot, zoom) in enumerate(test_data):
			print(f'Processing {idx+1}/{len(test_data)}', end='\r')
			xc = x.to(device)
			target = targets.to(device)
			yc = [yi.to(device) for yi in y]
			lossd = net.compute_loss(xc, target, yc)

			grasploss = lossd['loss']['grasp']
			classloss = lossd['loss']['class']
			loss = grasploss + classloss

			q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
														lossd['pred']['sin'], lossd['pred']['width'])

			s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
				no_grasps=args.n_grasps,
				grasp_width=width_img)
			if s:
				results['grasp']['correct'] += 1
			else:
				results['grasp']['failed'] += 1

			if args.jacquard_output:
				grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
				with open(jo_fn, 'a') as f:
					for g in grasps:
						f.write(test_data.dataset.get_jname(didx) + '\n')
						f.write(g.to_jacquard(scale=1024 / 300) + '\n')

			if args.vis:
				fig = visualise_output(xc[0], target[0], net, grasp_success=s)
				plt.show(fig)

	logging.info('IOU Results: %d/%d = %f' % (results['grasp']['correct'],
						results['grasp']['correct'] + results['grasp']['failed'],
						results['grasp']['correct'] / (results['grasp']['correct'] + results['grasp']['failed'])))

	if args.jacquard_output:
		print('Jacquard output saved to {}'.format(jo_fn))

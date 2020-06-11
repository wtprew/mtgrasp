import glob
import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

from utils.dataset_processing import grasp, image

from sklearn.model_selection import train_test_split

import torchvision.transforms.functional as TF

class JacquardKDataset(torch.utils.data.Dataset):
	"""
	Dataset wrapper for the Jacquard dataset
	"""
	def __init__(self, file_path, output_size=300,
				 random_rotate=False, random_zoom=False, include_rgb=True, include_depth=False,
				 shuffle=True, transform=None, ids=None):
		"""
		:param file_path: Jacquard Dataset directory.
		:param start: If splitting the dataset, split by this fraction [0, 1]
		"""

		self.file_path = file_path
		graspf = glob.glob(os.path.join(file_path, '*', '*', '*_grasps.txt'))
		graspf.sort()
		l = len(graspf)
		if l == 0:
			raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
		
		if ids is not None:
			ids = np.array(ids)
			graspf = [graspf[i] for i in ids]
		else:
			graspf = list(graspf)

		depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
		rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
		salf = [f.replace('grasps.txt', 'mask.png') for f in graspf]

		self.output_size = output_size
		self.random_rotate = random_rotate
		self.random_zoom = random_zoom
		self.rgb_files = rgbf
		self.depth_files = depthf
		self.grasp_files = graspf
		self.saliency_files = salf
		self.include_rgb = include_rgb
		self.include_depth = include_depth
		self.transform = transform

		if include_depth is False and include_rgb is False:
			raise ValueError('At least one of Depth or RGB must be specified.')

	def _get_crop_attrs(self, idx):
		gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
		center = gtbbs.center
		left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
		top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
		return center, left, top

	def get_gtbb(self, idx, rot=0, zoom=1.0):
		gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
		c = self.output_size//2
		gtbbs.rotate(rot, (c, c))
		gtbbs.zoom(zoom, (c, c))
		return gtbbs

	def get_depth(self, idx, rot=0, zoom=1.0):
		depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
		depth_img.rotate(rot)
		depth_img.normalise()
		depth_img.zoom(zoom)
		depth_img.resize((self.output_size, self.output_size))
		return depth_img.img

	def get_rgb(self, idx, rot=0, zoom=1.0, normalise=False):
		rgb_img = image.Image.from_file(self.rgb_files[idx])
		rgb_img.rotate(rot)
		rgb_img.zoom(zoom)
		rgb_img.resize((self.output_size, self.output_size))
		if normalise:
			rgb_img.normalise()
			rgb_img.img = rgb_img.img.transpose((2, 0, 1))
		return rgb_img.img

	def get_sal(self, idx, rot=0, zoom=1.0):
		sal_img = image.Image.from_file(self.saliency_files[idx])
		sal_img.rotate(rot)
		sal_img.zoom(zoom)
		sal_img.resize((self.output_size, self.output_size))
		return sal_img.img

	def get_jname(self, idx):
		return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])

	def __getitem__(self, idx):
		if self.random_rotate:
			rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
			rot = random.choice(rotations)
		else:
			rot = 0.0

		if self.random_zoom:
			zoom_factor = np.random.uniform(0.5, 1.0)
		else:
			zoom_factor = 1.0

		# Load the depth image
		if self.include_depth:
			depth_img = self.get_depth(idx, rot, zoom_factor)

		# Load the RGB image
		if self.include_rgb:
			rgb_img = self.get_rgb(idx, rot, zoom_factor, normalise=False)

		sal_img = self.get_sal(idx, rot, zoom_factor)

		# Load the grasps
		graspbbs = self.get_gtbb(idx, rot, zoom_factor)

		pos_img, ang_img, width_img = graspbbs.draw((self.output_size, self.output_size))
		width_img = np.clip(width_img, 0.0, 150.0)/150.0

		if self.include_depth and self.include_rgb:
			rgb_img = self.transform(rgb_img)
			depth_img = self.transform(depth_img)
			x = torch.cat([depth_img, rgb_img], 0)
		elif self.include_depth:
			x = self.transform(depth_img)
		elif self.include_rgb:
			x = self.transform(rgb_img)

		pos = self.transform(pos_img.astype(np.float32))
		cos = self.transform(np.cos(2*ang_img).astype(np.float32))
		sin = self.transform(np.sin(2*ang_img).astype(np.float32))
		width = self.transform(width_img.astype(np.float32))

		target = self.transform(sal_img)

		return x, target, (pos, cos, sin, width), idx, rot, zoom_factor

	def __len__(self):
		return len(self.grasp_files)

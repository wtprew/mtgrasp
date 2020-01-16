import glob
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

from utils.dataset_processing import grasp, image

# from sklearn.model_selection import train_test_split

class CornellCocoDataset(torch.utils.data.Dataset):
	"""
	Dataset wrapper for the Cornell dataset and coco annotations.
	"""
	def __init__(self, file_path, json, split = 0.8, output_size=300, 
				include_rgb=True, include_depth=False, **kwargs):
		"""
		:param file_path: Cornell Dataset directory.
		:param json: path to coco annotation file
		:param start: If splitting the dataset, split by this fraction [0, 1]
		:param kwargs: kwargs for GraspDatasetBase
		"""
		super(CornellCocoDataset, self).__init__(**kwargs)

		self.file_path = file_path
		self.coco = COCO(json)
		self.ids = list(self.coco.anns.keys())
		if len(self.ids) == 0:
			raise FileNotFoundError('No dataset files found. Check path: {}'.format(json))

		rgbf = []
		for k, v in self.coco.imgs.items():
			rgbf.append(os.path.join(file_path, self.coco.imgs[k]['file_name']))

		depthf = [f.replace('r.png', 'd.tiff') for f in rgbf]
		graspf = [f.replace('d.tiff', 'cpos.txt') for f in depthf]

		self.output_size = output_size
		self.rgb_files = rgbf
		self.depth_files = depthf
		self.grasp_files = graspf
		self.include_rgb = include_rgb
		self.include_depth = include_depth

		if include_depth is False and include_rgb is False:
			raise ValueError('At least one of Depth or RGB must be specified.')

	@staticmethod
	def numpy_to_torch(s):
		if len(s.shape) == 2:
			return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
		else:
			return torch.from_numpy(s.astype(np.float32))

	def _get_crop_attrs(self, idx):
		gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
		center = gtbbs.center
		left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
		top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
		return center, left, top

	def get_gtbb(self, idx):
		gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
		center, left, top = self._get_crop_attrs(idx)
		gtbbs.offset((-top, -left))
		return gtbbs

	def get_depth(self, idx):
		depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
		center, left, top = self._get_crop_attrs(idx)
		depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
		depth_img.normalise()
		depth_img.resize((self.output_size, self.output_size))
		return depth_img.img

	def get_rgb(self, idx, normalise=True):
		rgb_img = image.Image.from_file(self.rgb_files[idx])
		center, left, top = self._get_crop_attrs(idx)
		rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
		rgb_img.resize((self.output_size, self.output_size))
		if normalise:
			rgb_img.normalise()
			rgb_img.img = rgb_img.img.transpose((2, 0, 1))
		return rgb_img.img
	
	def get_anns(self, idx):
		ann_id = self.ids[idx]
		ann_bbox = self.coco.anns[ann_id]['bbox']
		ann_bbox = np.array(ann_bbox)
		target = self.coco.anns[ann_id]['category_id']
		target = np.array([target])
		return target, ann_bbox

	def __getitem__(self, idx):
		target, ann_bbox = self.get_anns(idx)
		
		# Load the depth image
		if self.include_depth:
			depth_img = self.get_depth(idx)

		# Load the RGB image
		if self.include_rgb:
			rgb_img = self.get_rgb(idx)

		# Load the grasps
		graspbbs = self.get_gtbb(idx)

		pos_img, ang_img, width_img = graspbbs.draw((self.output_size, self.output_size))
		width_img = np.clip(width_img, 0.0, 150.0)/150.0

		if self.include_depth and self.include_rgb:
			x = self.numpy_to_torch(
				np.concatenate(
					(np.expand_dims(depth_img, 0),
					 rgb_img),
					0
				)
			)
		elif self.include_depth:
			x = self.numpy_to_torch(depth_img)
		elif self.include_rgb:
			x = self.numpy_to_torch(rgb_img)

		pos = self.numpy_to_torch(pos_img)
		cos = self.numpy_to_torch(np.cos(2*ang_img))
		sin = self.numpy_to_torch(np.sin(2*ang_img))
		width = self.numpy_to_torch(width_img)
		
		target = self.numpy_to_torch(target)

		return x, (pos, cos, sin, width), idx, target

	def __len__(self):
		return len(self.ids)
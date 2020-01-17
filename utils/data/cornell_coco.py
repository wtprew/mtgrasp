import glob
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

from utils.dataset_processing import grasp, image

class CornellCocoDataset(torch.utils.data.Dataset):
	"""
	Dataset wrapper for the Cornell dataset and coco annotations.
	"""
	def __init__(self, file_path, annFile, start=0.0, end=1.0, output_size=300, 
				include_rgb=True, include_depth=False, transforms=None, **kwargs):

		self.file_path = file_path
		self.coco = COCO(annFile)
		self.ids = self.coco.getImgIds() # return list of ids
		if len(self.ids) == 0:
			raise FileNotFoundError('No dataset files found. Check path: {}'.format(json))
		
		self.cats = self.coco.loadCats(self.coco.getCatIds()) # get categories

		self.output_size = output_size
		self.include_rgb = include_rgb
		self.include_depth = include_depth

		if include_depth is False and include_rgb is False:
			raise ValueError('At least one of Depth or RGB must be specified.')

		self.transforms = transforms
	
	@staticmethod
	def numpy_to_torch(s):
		if len(s.shape) == 2:
			return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
		else:
			return torch.from_numpy(s.astype(np.float32))

	def __getitem__(self, idx):
		coco = self.coco
		img_id = self.ids[idx]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		target = coco.loadAnns(ann_ids)

		rgbpath = os.path.join(self.file_path, coco.loadImgs(img_id)[0]['file_name'])
		depthpath = rgbpath.replace('r.png', 'd.tiff')
		grasppath = rgbpath.replace('r.png', 'cpos.txt')

		if self.include_rgb:
			img = Image.open(rgbpath).convert('RGB')
	
		if self.include_depth:
			img = Image.open(depthpath)

		if self.transforms is not None:
			img = self.transforms(img)

		gtbbs = grasp.GraspRectangles.load_from_cornell_file(grasppath)
		pos_img, ang_img, width_img = gtbbs.draw((300,300))
		width_img = np.clip(width_img, 0.0, 150.0)/150.0

		pos = self.numpy_to_torch(pos_img)
		cos = self.numpy_to_torch(np.cos(2*ang_img))
		sin = self.numpy_to_torch(np.sin(2*ang_img))
		width = self.numpy_to_torch(width_img)

		targetcat = torch.tensor([target[0]['category_id']])
		targetbbox = torch.tensor([target[0]['bbox']])

		return img, (pos, cos, sin, width), targetcat, idx, targetbbox
	
	def __len__(self):
		return len(self.ids)
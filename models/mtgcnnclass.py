import torch
import torch.nn as nn
import torch.nn.functional as F


class MTGCNN(nn.Module):
	def __init__(self, input_channels=1, classes=9, shape_classes=4, material_classes=10, filter_sizes=None, l3_k_size=5, dilations=None):
		super().__init__()

		if filter_sizes is None:
			filter_sizes = [16,  # First set of convs
							16,  # Second set of convs
							32,  # Dilated convs
							16]  # Transpose Convs

		if dilations is None:
			dilations = [2, 4]

		self.features1 = nn.Sequential(
			# 4 conv layers.
			nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.features2 = nn.Sequential(
			# Dilated convolutions.
			nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
			nn.ReLU(inplace=True),

			# Output layers
			nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
		)

		self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.class_output = nn.Sequential(nn.Conv2d(filter_sizes[3], 1, kernel_size=1),
				nn.ReLU(inplace=True))

		self.linearlayers = nn.Sequential(
			nn.Linear(75*75, 512),
			nn.Dropout(p=0.5, inplace=True),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.Dropout(p=0.5, inplace=True),
			nn.ReLU(inplace=True),
		)

		self.fc = nn.Linear(256, classes)
		self.fc2 = nn.Linear(256, material_classes)
		self.fc3 = nn.Linear(256, shape_classes)

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.xavier_uniform_(m.weight, gain=1)

	def forward(self, x):
		y = self.features1(x)
		x = self.features2(y)

		pos_output = self.pos_output(x)
		cos_output = self.cos_output(x)
		sin_output = self.sin_output(x)
		width_output = self.width_output(x)

		class_output = self.class_output(y)
		# y = self.batch_norm(F.relu(class_output))
		y = torch.flatten(class_output, 1)
		y = self.linearlayers(y)
		classify = self.fc(y)
		shape = self.fc2(y)
		material = self.fc3(y)
		class_out = F.log_softmax(classify, dim=1)
		shape_out = F.log_softmax(shape, dim=1)
		material_out = F.log_softmax(material, dim=1)

		return pos_output, cos_output, sin_output, width_output, class_out, shape_out, material_out

	def compute_loss(self, xc, target, yc, grasp_weight=1.0, class_weight=1.0):
		"""
		xc: prediction from network
		yc: ground truth in same order as xc
		grasp_weight: weighting for loss function to be assigned to grasp
		class_weight: weighting for loss function to be assigned to class
		"""
		
		y_pos, y_cos, y_sin, y_width = yc
		y_class, y_material, y_shape = target['class_id'], target['material_id'], target['shape_id']
		pos_pred, cos_pred, sin_pred, width_pred, class_pred, mat_pred, shape_pred = self(xc)

		p_loss = F.mse_loss(pos_pred, y_pos)
		cos_loss = F.mse_loss(cos_pred, y_cos)
		sin_loss = F.mse_loss(sin_pred, y_sin)
		width_loss = F.mse_loss(width_pred, y_width)

		class_loss = F.nll_loss(class_pred, y_class)
		material_loss = F.nll_loss(mat_pred, y_material)
		shape_loss = F.nll_loss(shape_pred, y_shape)

		return {
			'loss': {
				'grasp': grasp_weight*(p_loss + cos_loss + sin_loss + width_loss),
				'class': class_weight*(class_loss + material_loss + shape_loss)
			},
			'losses': {
				'p_loss': p_loss,
				'cos_loss': cos_loss,
				'sin_loss': sin_loss,
				'width_loss': width_loss,
				'classify_loss': class_loss,
				'material_loss': material_loss,
				'shape_loss': shape_loss
			},
			'pred': {
				'pos': pos_pred,
				'cos': cos_pred,
				'sin': sin_pred,
				'width': width_pred,
				'class': class_pred,
				'material': mat_pred,
				'shape': shape_pred
			}
		}

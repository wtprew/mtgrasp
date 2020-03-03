import torch
import torch.nn as nn
import torch.nn.functional as F


class MTGCNNB(nn.Module):
	def __init__(self, input_channels=1, num_classes=76, filter_sizes=None, l3_k_size=5, dilations=None):
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

		self.fc = nn.Linear(256, num_classes)

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
		y = self.fc(y)
		class_out = F.log_softmax(y, dim=1)

		return pos_output, cos_output, sin_output, width_output, class_out

	def compute_loss(self, xc, target, yc, grasp_weight=1.0, class_weight=1.0):
		"""
		xc: prediction from network
		yc: ground truth in same order as xc
		grasp_weight: weighting for loss function to be assigned to grasp
		class_weight: weighting for loss function to be assigned to class
		"""
		
		y_pos, y_cos, y_sin, y_width = yc
		y_class = target
		pos_pred, cos_pred, sin_pred, width_pred, class_pred = self(xc)

		p_loss = F.mse_loss(pos_pred, y_pos)
		cos_loss = F.mse_loss(cos_pred, y_cos)
		sin_loss = F.mse_loss(sin_pred, y_sin)
		width_loss = F.mse_loss(width_pred, y_width)

		class_loss = F.nll_loss(class_pred, y_class)

		return {
			'loss': {
				'grasp': grasp_weight*(p_loss + cos_loss + sin_loss + width_loss),
				'class': class_weight*(class_loss)
			},
			'losses': {
				'p_loss': p_loss,
				'cos_loss': cos_loss,
				'sin_loss': sin_loss,
				'width_loss': width_loss,
				'class_loss': class_loss
			},
			'pred': {
				'pos': pos_pred,
				'cos': cos_pred,
				'sin': sin_pred,
				'width': width_pred,
				'class': class_pred
			}
		}

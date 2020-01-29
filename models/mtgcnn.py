import torch
import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class MTGCNN(nn.Module):
	"""
	extension of the GG-CNN with a classification branch
	"""
	def __init__(self, input_channels=1, num_classes=70):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
		self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
		self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
		self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
		self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
		self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

		self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

		self.class_conv = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.linear1 = nn.Linear(300*300, 512)
		self.linear2 = nn.Linear(512, 256)
		self.class_output = nn.Linear(256, num_classes)

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.xavier_uniform_(m.weight, gain=1)

	def forward(self, x):
		#shared network
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.convt1(x))
		x = F.relu(self.convt2(x))
		x = F.relu(self.convt3(x))

		pos_output = self.pos_output(x)
		cos_output = self.cos_output(x)
		sin_output = self.sin_output(x)
		width_output = self.width_output(x)

		#linear layers for classification

		y = self.class_conv(x)
		y = torch.flatten(y, 1)
		y = F.relu(self.linear1(y))
		y = F.relu(self.linear2(y))
		class_out = self.class_output(y)
		class_out = F.log_softmax(y, dim=0)

		return pos_output, cos_output, sin_output, width_output, class_out

	def compute_loss(self, xc, yc, grasp_weight=1.0, class_weight=1.0):
		"""
		xc: prediction from network
		yc: ground truth in same order as xc
		grasp_weight: weighting for loss function to be assigned to grasp
		class_weight: weighting for loss function to be assigned to class
		"""
		
		y_pos, y_cos, y_sin, y_width, y_class = yc
		pos_pred, cos_pred, sin_pred, width_pred, class_pred = self(xc)

		p_loss = F.mse_loss(pos_pred, y_pos)
		cos_loss = F.mse_loss(cos_pred, y_cos)
		sin_loss = F.mse_loss(sin_pred, y_sin)
		width_loss = F.mse_loss(width_pred, y_width)

		class_loss = F.nll_loss(class_pred, y_class.squeeze(1))

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

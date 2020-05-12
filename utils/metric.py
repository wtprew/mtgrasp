import numpy as np

class AvgMeter(object):
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
def cal_pr_mae_meanf(prediction, gt):
	assert prediction.dtype == np.uint8
	assert gt.dtype == np.uint8
	assert prediction.shape == gt.shape
	
	if prediction.max() == prediction.min():
		prediction = prediction / 255
	else:
		prediction = ((prediction - prediction.min()) /
					  (prediction.max() - prediction.min()))
	hard_gt = np.zeros_like(gt)
	hard_gt[gt > 128] = 1
	
	# MAE 
	mae = np.mean(np.abs(prediction - hard_gt))
	
	# MeanF 
	threshold_fm = 2 * prediction.mean()
	if threshold_fm > 1:
		threshold_fm = 1
	binary = np.zeros_like(prediction)
	binary[prediction >= threshold_fm] = 1
	tp = (binary * hard_gt).sum()
	if tp == 0:
		meanf = 0
	else:
		pre = tp / binary.sum()
		rec = tp / hard_gt.sum()
		meanf = 1.3 * pre * rec / (0.3 * pre + rec)
	
	# PR curve 
	t = np.sum(hard_gt)
	precision, recall = [], []
	for threshold in range(256):
		threshold = threshold / 255.
		hard_prediction = np.zeros_like(prediction)
		hard_prediction[prediction >= threshold] = 1
		
		tp = np.sum(hard_prediction * hard_gt)
		p = np.sum(hard_prediction)
		if tp == 0:
			precision.append(0)
			recall.append(0)
		else:
			precision.append(tp / p)
			recall.append(tp / t)
	
	return precision, recall, mae, meanf


# MaxF 
def cal_maxf(ps, rs):
	assert len(ps) == 256
	assert len(rs) == 256
	maxf = []
	for p, r in zip(ps, rs):
		if p == 0 or r == 0:
			maxf.append(0)
		else:
			maxf.append(1.3 * p * r / (0.3 * p + r))
	
	return max(maxf)

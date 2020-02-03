import matplotlib.pyplot as plt
import numpy as np
import sklearn
import itertools


def plot_confusion_matrix(pred, target, class_names):
	"""
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
	cm (array, shape = [n, n]): a confusion matrix of integer classes
	class_names (array, shape = [n]): String names of the integer classes
	"""

	cm = sklearn.metrics.confusion_matrix(target, pred)

	figure = plt.figure(figsize=(8, 8), dpi=320, facecolor='w', edgecolor='k')
	ax = figure.add_subplot(1,1,1)
	ax.imshow(cm, cmap='Oranges')
	plt.title("Confusion matrix")
	ax.colorbar()
	tick_marks = np.arange(len(class_names))
	ax.xticks(tick_marks, class_names, rotation=90)
	ax.yticks(tick_marks, class_names)

	# Normalize the confusion matrix.
	cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

	# Use white text if squares are dark; otherwise black.
	threshold = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > threshold else "black"
		plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	ax.ylabel('True label')
	ax.xlabel('Predicted label')

	return figure

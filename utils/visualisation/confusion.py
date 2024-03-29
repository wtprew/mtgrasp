import itertools
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(correct_labels, predict_labels, classes, title='Confusion matrix', normalize=False):
	''' 
	Parameters:
	    correct_labels                  : These are your true classification categories.
	    predict_labels                  : These are you predicted classification categories
	    labels                          : This is a lit of labels which will be used to display the axix labels
	    title='Confusion matrix'        : Title for your matrix
	'''
	tick_marks = np.arange(len(classes))

	cm = confusion_matrix(correct_labels, predict_labels, labels=tick_marks)
	if normalize:
	    cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
	    cm = np.nan_to_num(cm, copy=True)
	    cm = cm.astype('int')

	np.set_printoptions(precision=2)

	fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)
	im = ax.imshow(cm, cmap='Oranges')

	ax.set_xlabel('Predicted', fontsize=4)
	ax.set_xticks(tick_marks)
	c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()

	ax.set_ylabel('True Label', fontsize=4)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize=4, va ='center')
	ax.yaxis.set_label_position('left')
	ax.yaxis.tick_left()

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")

	return fig

def count_elements(dataset, classes, key) -> dict:
	hist = {}
	for i in dataset.targets:
		cat = i['category_id']
		if key == 'supercategory_id':
			name = dataset.coco.loadCats(cat)[0][str(key[:-3])]
		else:
			name = classes[cat - 1]
		hist[name] = hist.get(name, 0) + 1
	return hist

def count_list(l, classes) -> dict:
	hist = {}
	for i in l:
		name = classes[i]
		hist[name] = hist.get(name, 0) + 1
	return hist

def histogram_plot(freq):
	tick_marks = np.arange(len(freq.keys()))

	fig = plt.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)

	ax.bar(freq.keys(), freq.values())

	ax.set_xlabel('Classes', fontsize=4)
	ax.set_xticks(tick_marks)
	c = ax.set_xticklabels(freq.keys(), fontsize=4, rotation=-90,  ha='center')
	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()

	ax.set_ylabel('Count', fontsize=4)
	ax.yaxis.tick_left()

	return fig
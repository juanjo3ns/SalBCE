import os
import sys
import cv2

import numpy as np
from IPython import embed
from PIL import Image
from random import randint
from scipy import ndimage

import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

size = (192, 256)
# BGR MODE
mean = [103.939, 116.779, 123.68]
PATH_SALICON = "/home/dataset/SALICON/"

def imageProcessing(image, saliency):


	image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
	saliency = cv2.resize(saliency, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

	# remove mean value
	image -= mean
	augmentation = randint(0,3)
	if augmentation == 0:
		image = image[:,::-1,:]
		saliency = saliency[:,::-1]
	elif augmentation == 1:
		image = image[::-1,:,:]
		saliency = saliency[::-1,:]
	elif augmentation == 2:
		image = ndimage.rotate(image, 45)
		saliency = ndimage.rotate(saliency, 45)
		sqr = image.shape[0]
		start1 = int((sqr-192)/2)+1
		end1 = sqr-int((sqr-192)/2)
		start2 = int((sqr-256)/2)+1
		end2 = sqr-int((sqr-256)/2)
		image = image[start1:end1, start2:end2,:]
		saliency = saliency[start1:end1, start2:end2]
	# convert to torch Tensor
	image = np.ascontiguousarray(image)
	saliency = np.ascontiguousarray(saliency)

	image = torch.FloatTensor(image)

	# swap channel dimensions
	image = image.permute(2,0,1)
	return image,saliency

class SALICON(Dataset):
	def __init__(self, mode='train', return_path=False, N=None):
		global PATH_SALICON
		self.size = (192, 256)
		# MEAN IN BGR MODE
		self.mean = [103.939, 116.779, 123.68]
		# self.mean = [123.68, 116.779, 103.939]
		self.path_dataset = PATH_SALICON
		self.path_images = os.path.join(self.path_dataset,'image', 'images')
		self.path_saliency = os.path.join(self.path_dataset, 'maps', mode)
		self.return_path = return_path

		# get list images
		list_names = os.listdir( os.path.join(self.path_dataset, 'fixations', mode) )
		list_names = np.array([n.split('.')[0] for n in list_names])
		self.list_names = list_names

		if N is not None:
			self.list_names = list_names[:N]
		embed()
		print("Init dataset in mode {}".format(mode))
		print("\t total of {} images.".format(self.list_names.shape[0]))

	def __len__(self):
		return self.list_names.shape[0]

	def __getitem__(self, index):

		# Image and saliency map paths
		rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
		sal_path = os.path.join(self.path_saliency, self.list_names[index]+'.png')

		image = cv2.imread(rgb_ima)
		saliency = cv2.imread(sal_path, 0)
		return imageProcessing(image, saliency)

if __name__ == '__main__':
	s = SALICON(mode='val', N=100)

	image, saliency = s[0]

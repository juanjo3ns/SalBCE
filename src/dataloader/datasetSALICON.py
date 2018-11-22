import os
import sys
import cv2

import numpy as np
from IPython import embed
from PIL import Image

import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PATH_SALICON = "/home/dataset/SALICON/"

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

		print("Init dataset in mode {}".format(mode))
		print("\t total of {} images.".format(list_names.shape[0]))

	def __len__(self):
		return self.list_names.shape[0]

	def __getitem__(self, index):

		# Image and saliency map paths
		rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
		sal_path = os.path.join(self.path_saliency, self.list_names[index]+'.png')
		# Load image and saliency map
		# image = Image.open(rgb_ima)
		# saliency = Image.open(sal_path).convert('LA')
		image = cv2.imread(rgb_ima)
		saliency = cv2.imread(sal_path, 0)

		return imageProcessing(image, saliency)


	def imageProcessing(image, saliency):


		image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

		# convert to foat
		image = image.astype(np.float32)
		saliency = saliency.astype(np.float32)

		# remove mean value
		image -= self.mean
		embed()
		if random.randint(0,2) == 0:
			[start:end:step,start:end:step,start:end:step]
			image = image[:,::-1,:]
		# convert to torch Tensor
		image = torch.FloatTensor(image)

		# swap channel dimensions
		image = image.permute(2,0,1)
		return image,saliency

if __name__ == '__main__':
	s = SALICON(mode='val', N=100)

	image, saliency = s[0]

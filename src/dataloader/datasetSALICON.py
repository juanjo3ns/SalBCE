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

# from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, VerticalFlip

# constants
PATH_SALICON = "/home/dataset/"
size = (192, 256)
mean = [103.939, 116.779, 123.68]

class SALICON(Dataset):
	def __init__(self, mode='train', transform=None, return_path=False, N=None):
		global PATH_SALICON
		self.size = (192, 256)
		self.mean = [103.939, 116.779, 123.68]
		self.path_dataset = PATH_SALICON
		self.path_images = os.path.join(self.path_dataset,'image', 'images')
		self.path_saliency = os.path.join(self.path_dataset, 'maps', mode)
		self.transform = transform
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
		# set path
		ima_name = self.list_names[index]+'.jpg'
		rgb_ima = os.path.join(self.path_images, ima_name)
		# image = cv2.imread(rgb_ima)
		sal_name = self.list_names[index]+'.png'
		sal_path = os.path.join(self.path_saliency, sal_name)
		# saliency = cv2.imread(os.path.join(self.path_saliency, ima_name), 0)
		image = Image.open(rgb_ima)
		saliency = Image.open(sal_path)
		if self.transform:
			image = self.transform(image)
			saliency = self.transform(saliency)
		#
		# if self.transform:
		# 	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		#
		# 	image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		# 	saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		#
		# 	image = image.astype(np.float32)
		# 	saliency = saliency.astype(np.float32)
		#
		# 	image -= self.mean
		# 	# augmented = self.transform(image=image)
		# 	# image = augmented['image']
		# 	augmented_s = self.transform(image=saliency)
		# 	saliency = augmented_s['image']
		# 	embed()
		# 	image = image.permute(2,0,1)

		#apply transformation
		# if self.transform is not None:
		# 	 # reshape
		# 	 image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		# 	 saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		#
		# 	 # convert to foat
		# 	 image = image.astype(np.float32)
		# 	 saliency = saliency.astype(np.float32)
		#
		# 	 # remove mean value
		# 	 image -= self.mean
		#
		# 	 # convert to torch Tensor
		# 	 image = torch.FloatTensor(image)
		#
		# 	 # swap channel dimensions
		# 	 image = image.permute(2,0,1)


		if self.return_path:
			return image, saliency, rgb_ima
		else:
			return image, saliency




if __name__ == '__main__':
	s = SALICON(mode='val', transform=torchvision_transform, N=100)

	image, saliency = s[0]

import os
import sys
import cv2

import numpy as np
from IPython import embed

import matplotlib.pylab as plt
from random import randint
from scipy import ndimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATH = "/home/dataset/"
TRAIN = 'train'
VAL = 'val'
numTraining = 456
numValidation = 41

class LEDOV(Dataset):
	def __init__(self, mode='train',transformation=None, return_path=False, N=None):
		global PATH_DHF1K
		self.size = (192, 256)
		self.mean = [103.939, 116.779, 123.68]
		self.path_dataset = PATH
		self.path_images = os.path.join(self.path_dataset, 'ledov_frames')
		self.path_saliency = os.path.join(self.path_dataset, 'ledov_gt')

		self.transformation = transformation
		self.return_path = return_path
		videos = os.listdir(self.path_images)

		list_names = []
		if mode == TRAIN:
			for v in videos[:numTraining]:
				for t in os.listdir(os.path.join(self.path_images, v)):
					list_names.append(v+t)
		elif mode == VAL:
			for v in videos[numTraining:(numTraining+numValidation)]:
				for t in os.listdir(os.path.join(self.path_images, v)):
					list_names.append(v+t)
		list_names = np.array([n.split('.')[0] for n in list_names])
		self.list_names = list_names

		if N is not None:
			self.list_names = list_names[:N]
		print("Total of {} images.".format(self.list_names.shape[0]))

	def __len__(self):
		return self.list_names.shape[0]

	def __getitem__(self, index):
		# set path
		ima_name = self.list_names[index][-4:]+'.jpg' # the last 4 numbers belong to the image name
		vid = self.list_names[index][:len(self.list_names[index])-4] #  the others belong to the name of video

		rgb_ima = os.path.join(self.path_images,vid, ima_name)
		image = cv2.imread(rgb_ima)
		ima_name = self.list_names[index][-4:]+'.png'
		saliency = cv2.imread(os.path.join(self.path_saliency, vid,'GT','map', str(int(ima_name.split('.')[0])) + '.png'), 0)

		# apply transformation
		if self.transformation is not None:
			# reshape
			image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
			saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

			# convert to foat
			image = image.astype(np.float32)
			saliency = saliency.astype(np.float32)

			# remove mean value
			image -= self.mean
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

			# convert to torch Tensor
			image = torch.FloatTensor(image)

			# swap channel dimensions
			image = image.permute(2,0,1)


		if self.return_path:
			return image, saliency, rgb_ima
		else:
			return image, saliency

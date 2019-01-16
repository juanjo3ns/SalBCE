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
from utils.load_flow import readFlow
from torchvision import transforms

PATH_DHF1K = "/home/dataset/DHF1K/"
TRAIN = 'train'
VAL = 'val'

def augmentData(image,saliency):
	augmentation = randint(0,3)
	augmentation = 2
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
	return image, saliency
def img_name(name, num):
	num_image = int(name.split('.')[0])
	if num_image < num:
		return '{:04d}.png'.format(num)
	else: return '{:04d}.png'.format(int(num_image/num)*num)

class DHF1K(Dataset):
	def __init__(self, mode='train',transformation=False, return_path=False, N=None, depth=False, d_augm=False, coord=False):
		global PATH_DHF1K
		self.size = (192, 256)
		self.mean = [103.939, 116.779, 123.68] #BGR
		self.path_dataset = PATH_DHF1K
		self.path_images = os.path.join(self.path_dataset, 'dhf1k_frames')
		self.path_saliency = os.path.join(self.path_dataset, 'dhf1k_gt')
		self.path_depth = os.path.join(self.path_dataset, 'dhf1k_depth')
		self.depth = depth
		self.d_augm = d_augm
		self.coord = coord
		self.transformation = transformation
		self.return_path = return_path

		list_names = []
		if mode == TRAIN:
			for v in range(1,601):
				for t in os.listdir(os.path.join(self.path_dataset,'dhf1k_gt',str(v),'maps')):
					list_names.append(str(v)+t)
		elif mode == VAL:
			for v in range(601,701):
				for t in os.listdir(os.path.join(self.path_dataset,'dhf1k_gt',str(v),'maps')):
					list_names.append(str(v)+t)
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
		vid = int(self.list_names[index][:len(self.list_names[index])-4]) #  the others belong to the number of video

		rgb_ima = os.path.join(self.path_images,'{:03}'.format(vid), ima_name)
		image = cv2.imread(rgb_ima)
		ima_name = self.list_names[index][-4:]+'.png'
		saliency = cv2.imread(os.path.join(self.path_saliency, str(vid), 'maps', ima_name), 0)
		# apply transformation
		if self.transformation:
			# reshape
			image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
			saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
			# convert to foat
			image = image.astype(np.float32)
			saliency = saliency.astype(np.float32)

			# remove mean value
			image -= self.mean

			# The order we add [DEPTH, FLOW, COORD, DATA AUGMENTATION] matters!
			# Add 1 channel with image depth if required
			if self.depth:
				ima_name = img_name(ima_name,10)
				depth = cv2.imread(os.path.join(self.path_depth,str(vid),ima_name), 0)
				depth = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
				depth = depth.astype(np.float32)
				depth = np.expand_dims(depth, axis=2)
				image = np.dstack((image,depth))

			# Add 2 channel with image optical flow if required
			# [NOT COMPLETED BECAUSE OF SPACE MEMORY ISSUES]
			# if self.flow:
			# 	ima_name = img_name(ima_name,11)



			#Data augmentation if required
			if self.d_augm and randint(0,1)==0:
				image, saliency = augmentData(image,saliency)


			# Add 2 channel with coordinates
			if self.coord:
				c1 = np.empty(shape=(self.size[0], self.size[1]))
				c2 = np.empty(shape=(self.size[0], self.size[1]))
				for i in range(0,self.size[0]):
					c1[i,:]= i
				for i in range(0,self.size[1]):
					c2[:,i]= i
				c1 = c1.astype(np.float32)
				c2 = c2.astype(np.float32)
				c1 = np.expand_dims(c1, axis=2)
				c2 = np.expand_dims(c2, axis=2)
				image = np.dstack((image,c1))
				image = np.dstack((image,c2))



			# convert to torch Tensor
			image = torch.FloatTensor(image)

			# swap channel dimensions
			image = image.permute(2,0,1)



		if self.return_path:
			return image, saliency, rgb_ima
		else:
			return image, saliency


if __name__ == '__main__':
	ds = DHF1K(mode='val', transformation=True, depth=False, d_augm=True, coord=True)
	image, saliency = ds[0]

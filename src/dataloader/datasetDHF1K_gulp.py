import os
import sys
import cv2

import numpy as np
from IPython import embed

import matplotlib.pylab as plt
from random import randint
from scipy import ndimage
import torch


PATH_DHF1K = "/home/dataset/DHF1K/"
TRAIN = 'train'
VAL = 'val'

def augmentData(image,saliency):
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
	return image, saliency

class DHF1K(Dataset):
	def __init__(self, mode='train',transformation=False, return_path=False, N=None, depth=False, d_augm=False):
		global PATH_DHF1K
		self.size = (192, 256)
		self.mean = [103.939, 116.779, 123.68] #BGR
		self.path_dataset = PATH_DHF1K
		self.path_images = os.path.join(self.path_dataset, 'dhf1k_frames')
		self.path_saliency = os.path.join(self.path_dataset, 'dhf1k_gt')
		self.path_depth = os.path.join(self.path_dataset, 'dhf1k_depth')
		self.depth = depth
		self.d_augm = d_augm
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

			# Add 4 channel with image depth if required
			if self.depth:
				num_image = int(ima_name.split('.')[0])
				if num_image < 10:
					ima_name = '0010.png'
				else: ima_name = '{:04d}.png'.format(int(num_image/10)*10)
				depth = cv2.imread(os.path.join(self.path_depth,str(vid),ima_name), 0)
				depth = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
				depth = depth.astype(np.float32)
				depth = np.expand_dims(depth, axis=2)
				image = np.dstack((image,depth))

			#Data augmentation if required
			if self.d_augm:
				image, saliency = augmentData(image,saliency)

			# convert to torch Tensor
			image = torch.FloatTensor(image)

			# swap channel dimensions
			image = image.permute(2,0,1)



		if self.return_path:
			return image, saliency, rgb_ima
		else:
			return image, saliency


if __name__ == '__main__':
	from gulpio.dataset import GulpImageDataset
	from gulpio.loader import DataLoader
	ds_train = DHF1K(mode=TRAIN, transformation=True, depth=DEPTH, d_augm=AUGMENT)
	ds_validate = DHF1K(mode=VAL, transformation=True, depth=DEPTH, d_augm=AUGMENT)

	# Dataloaders
	dataloader = {
		TRAIN: DataLoader(ds_train, batch_size=batch_size,
								shuffle=True, num_workers=2),
		VAL: DataLoader(ds_validate, batch_size=batch_size,
								shuffle=False, num_workers=2)
	}
	for data, label in dataloader:

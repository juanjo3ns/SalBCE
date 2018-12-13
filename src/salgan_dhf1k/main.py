import sys
import torch
#from utils.salgan_generator import create_model
from utils.salgan_generator_bn import create_model
from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

import numpy as np
import os
import cv2

import matplotlib.pylab as plt
from IPython import embed

PATH_PYTORCH_WEIGHTS = '../trained_models/salcoord_dhf1k_batchnorm/models/best.pt'
INPUT_PATH = '/home/dataset/DHF1K/dhf1k_frames/'
OUTPUT_PATH = '/home/saliency_maps/salcoord_dhf1k_batchnorm'
DEPTH_PATH = '/home/dataset/DHF1K/dhf1k_depth/'

USE_GPU=True


def main():
	"""
	Runs pytorch-SalGAN on a sample images

	"""
	DEPTH = False
	COORD = True
	# create output file
	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	# init model with pre-trained weights
	if DEPTH:
		model = create_model(4)
	elif COORD:
		model = create_model(5)
	else: model = create_model(3)

	model.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS)['state_dict'])
	model.eval()


	# if GPU is enabled
	if USE_GPU:
		model.cuda()

	# load and preprocess images in folder
	for y in range(601,701):
		if not os.path.exists(os.path.join(OUTPUT_PATH,str(y))):
			os.makedirs(os.path.join(OUTPUT_PATH,str(y)))
		for n, name in enumerate(os.listdir(os.path.join(INPUT_PATH,'{:03d}'.format(y)))):
			filename = os.path.join(INPUT_PATH,'{:03d}'.format(y), name)
			image_tensor, image_size = load_image(filename)
			if DEPTH:
				num_image = int(name.split('.')[0])
				if num_image < 10:
					ima_name = '0010.png'
				else: ima_name = '{:04d}.png'.format(int(num_image/10)*10)
				depth = cv2.imread(os.path.join(DEPTH_PATH,str(y),ima_name), 0)
				depth = cv2.resize(depth, (256, 192), interpolation=cv2.INTER_AREA)
				depth = depth.astype(np.float32)
				depth = np.expand_dims(depth, axis=0)
				depth = torch.FloatTensor(depth)
				image_tensor = torch.cat([image_tensor,depth],0)
			if COORD:
				c1 = np.empty(shape=(192, 256))
				c2 = np.empty(shape=(192, 256))
				for i in range(0,192):
					c1[i,:]= i
				for i in range(0,256):
					c2[:,i]= i
				c1 = c1.astype(np.float32)
				c2 = c2.astype(np.float32)
				c1 = np.expand_dims(c1, axis=0)
				c2 = np.expand_dims(c2, axis=0)
				c1 = torch.FloatTensor(c1)
				c2 = torch.FloatTensor(c2)
				image_tensor = torch.cat([image_tensor,c1],0)
				image_tensor = torch.cat([image_tensor,c2],0)
			if USE_GPU:
				image_tensor = image_tensor.cuda()

			# run model inference
			prediction = model.forward(image_tensor[None, ...]) # add extra batch dimension

			# get result to cpu and squeeze dimensions
			if USE_GPU:
				prediction = prediction.squeeze().data.cpu().numpy()
			else:
				prediction = prediction.squeeze().data.numpy()

			# postprocess
			# first normalize [0,1]
			prediction = normalize_map(prediction)
			saliency = postprocess_prediction(prediction, image_size)
			saliency = normalize_map(saliency)
			saliency *= 255
			saliency = saliency.astype(np.uint8)
			# save saliency

			cv2.imwrite(os.path.join(OUTPUT_PATH,str(y),name), saliency)
			print("Processed image {} from video {}".format(n,y), end="\r")
			sys.stdout.flush()
	print("\n")

if __name__ == '__main__':
	main()

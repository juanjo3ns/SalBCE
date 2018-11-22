import sys
import torch
from utils.salgan_generator import create_model
from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

import numpy as np
import os
import cv2

import matplotlib.pylab as plt
from IPython import embed

PATH_PYTORCH_WEIGHTS = os.path.join(os.environ['PYTHONPATH'],'trained_models/salgan_salicon_3epochs/models/best.pt')
INPUT_PATH = '/home/dataset/LEDOV/ledov_frames/'
OUTPUT_PATH = '/home/saliency_maps/salgan_ledov_3epochssalicon/'
USE_GPU=True
numTraining = 456
numValidation = 41

def main():
	"""
	Runs pytorch-SalGAN on a sample images

	"""
	# create output file
	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	# init model with pre-trained weights
	model = create_model()

	model.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS)['state_dict'])
	model.eval()


	# if GPU is enabled
	if USE_GPU:
		model.cuda()
	videos = os.listdir(INPUT_PATH)
	# load and preprocess images in folder
	for y in videos[numTraining:(numTraining+numValidation)]:
		if not os.path.exists(os.path.join(OUTPUT_PATH,y)):
			os.makedirs(os.path.join(OUTPUT_PATH,y))
			for i, name in enumerate(os.listdir(os.path.join(INPUT_PATH,y))):
				filename = os.path.join(INPUT_PATH,y,'{:04d}.jpg'.format(i+1))
				image_tensor, image_size = load_image(filename)

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
				print("Processed image {} from video {}".format(i+1,y), end="\r")
				sys.stdout.flush()

if __name__ == '__main__':
	main()

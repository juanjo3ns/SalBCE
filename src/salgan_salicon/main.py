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

PATH_PYTORCH_WEIGHTS = '/home/code/trained_models/salgan_salicon_dataaugmentation/models/best.pt'
INPUT_PATH = '/home/dataset/image/images'
#save saliency with this format: /model_dataset_configuration
OUTPUT_PATH = '/home/saliency_maps/salgan_salicon_dataaugmentation/'

USE_GPU=True


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

	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)
	# load and preprocess images in folder
	for img in os.listdir(INPUT_PATH):
		if 'test' in img:
			filename = os.path.join(INPUT_PATH, img)
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

			cv2.imwrite(os.path.join(OUTPUT_PATH,img), saliency)
			print("Processed image {} ".format(img), end="\r")
			sys.stdout.flush()

if __name__ == '__main__':
	main()

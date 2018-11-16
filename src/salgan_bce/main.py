import sys
import torch
import ..utils.salgan_generator
from ..utils.salgan_utils import load_image, postprocess_prediction
from ..utils.salgan_utils import normalize_map

import numpy as np
import os
import cv2

import matplotlib.pylab as plt
from IPython import embed

# parameters for demo inference=================================================
# PATH_PYTORCH_WEIGHTS = 'model_weights/gen_model.pt'
# PATH_SAMPLE_IMAGES = 'sample_images'
# PATH_SAMPLE_SALIENCY = 'sample_saliency'
PATH_PYTORCH_WEIGHTS = 'model_weights/gen_model.pt'
INPUT_PATH = '/salgan_pytorch/data/dhf1k_frames/'
OUTPUT_PATH = '/salgan_pytorch/data/salganbaseline/'
USE_GPU=True


def main():
	"""
	Runs pytorch-SalGAN on a sample images

	"""
	# create output file
	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	# init model with pre-trained weights
	model = salgan_generator.create_model()

	model.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS))
	model.eval()


	# if GPU is enabled
	if USE_GPU:
		model.cuda()

	# load and preprocess images in folder
	for y in range(601,701):
		if not os.path.exists(os.path.join(OUTPUT_PATH,str(y))):
			os.makedirs(os.path.join(OUTPUT_PATH,str(y)))
		for i, name in enumerate(os.listdir(os.path.join(INPUT_PATH,'{:03d}'.format(y)))):
			filename = os.path.join(INPUT_PATH,'{:03d}'.format(y), name)
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
			print("Processed image {} from video {}".format(i,y), end="\r")
			sys.stdout.flush()

if __name__ == '__main__':
	main()

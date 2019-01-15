
import sys
import torch
#from utils.salgan_generator import create_model
from utils.salgan_generator import create_model, add_bn
from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

from utils.sendTelegram import send

import cv2
import os
import random
import numpy as np

from IPython import embed

PATH_PYTORCH_WEIGHTS = '../trained_models/salgan_baseline.pt'
USE_GPU=True
SALGAN_RESIZE = (192, 256) # H, W
size = None
bgr_mean=[103.939, 116.779, 123.68]
model = create_model(3)
model.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS)['state_dict'])
model.eval()

# if GPU is enabled
if USE_GPU:
	model.cuda()


filename = '/home/dataset/DHF1K/dhf1k_frames/199/0199.jpg'
image = cv2.imread(filename) # BGR format
image2 = image[0:160,0:320]
H, W, C = image.shape
if size is None:
    size = SALGAN_RESIZE

image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
image = image.astype(np.float32)
image2 = cv2.resize(image2, (size[1], size[0]), interpolation=cv2.INTER_AREA)
image2 = image2.astype(np.float32)

bgr_mean=np.array(bgr_mean)
image -= bgr_mean
image2 -= bgr_mean

# convert to torch Tensor
image = torch.FloatTensor(image)
image2 = torch.FloatTensor(image2)

# swap channel dimensions
image = image.permute(2,0,1)
image2 = image2.permute(2,0,1)


if USE_GPU:
	image = image.cuda()
	image2 = image2.cuda()

for img, name in zip([image,image2],["nocrop.png","crop.png"]):
	# run model inference
	prediction = model.forward(img[None, ...]) # add extra batch dimension

	# get result to cpu and squeeze dimensions
	if USE_GPU:
		prediction = prediction.squeeze().data.cpu().numpy()
	else:
		prediction = prediction.squeeze().data.numpy()

	# postprocess
	# first normalize [0,1]
	prediction = normalize_map(prediction)
	saliency = postprocess_prediction(prediction, (320,640))
	saliency = normalize_map(saliency)
	saliency *= 255
	cv2.imwrite(os.path.join("/home/saliency_maps/", name),saliency)

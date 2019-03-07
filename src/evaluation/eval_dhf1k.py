
#-------------------------------------------------------------------------------------#
	# This script is gonna be used when we want to compute
	# the saliency metrics of all the validation set
	# where we create the model from a pretrained weights
#-------------------------------------------------------------------------------------#

import sys
import torch
#from utils.salgan_generator import create_model
from utils.salgan_generator import create_model, add_bn
from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

from utils.sendTelegram import send

from evaluation.metrics_functions import AUC_Judd, AUC_shuffled, CC, NSS, SIM
import cv2
import os
import random
import numpy as np
import datetime
import time

from IPython import embed

USE_GPU = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "--model", type=str,
			help="""Weight's path and folder name for saliency maps.""")
parser.add_argument("--depth", default=False, type=bool,
			help="""Enable 4th channel with depth""")
parser.add_argument("--coord", default=False, type=bool,
			help="""Enable coordconv""")
parser.add_argument("--save", default=False, type=bool,
			help="""Save inference images""")
args = parser.parse_args()

MODEL = args.model
SAVE = args.save
DEPTH = args.depth
COORD = args.coord

PATH_PYTORCH_WEIGHTS = '../trained_models/batch12_/'+ MODEL +'/models/best.pt'
# PATH_PYTORCH_WEIGHTS = '../trained_models/salgan_baseline.pt'
DHF1K_PATH = '/home/dataset/DHF1K/'
INPUT_PATH = os.path.join(DHF1K_PATH,'dhf1k_frames/')

DEPTH_PATH = os.path.join(DHF1K_PATH,'dhf1k_depth/')
GT_PATH = os.path.join(DHF1K_PATH, 'dhf1k_gt/')
OUTPUT_PATH = '/home/saliency_maps/' + MODEL

start_total = time.time()

final_metric_list = []

def construct_other():
	other_map = np.zeros((360,640))
	for i in range(0,50):
		random_video = random.randint(601, len(os.listdir(GT_PATH)))
		other_map += cv2.imread(os.path.join(GT_PATH, str(random_video),'fixation', '{:04d}.png'.format(random.randint(1, len(os.listdir(os.path.join(GT_PATH, str(random_video),'fixation')))))),cv2.IMREAD_GRAYSCALE)
		other_map = np.clip(other_map,0,255)
	return other_map

def inference(y):
	predictions = []
	files = sorted(os.listdir(os.path.join(INPUT_PATH,'{:03d}'.format(y))), key = lambda x: int(x.split(".")[0]))
	for file in files:
		filename = os.path.join(INPUT_PATH,'{:03d}'.format(y), file)
		image_tensor, image_size = load_image(filename)
		if DEPTH:
			num_image = int(file.split('.')[0])
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
			for i in range(0,256):
				c1[:,i] = np.linspace(-1,1,192)
			for i in range(0,192):
				c2[i,:]= np.linspace(-1,1,256)
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
		predictions.append(saliency)
	return predictions

def inner_worker(n, pack, y):
	gt, prediction = pack
	if SAVE:
		cv2.imwrite(os.path.join(OUTPUT_PATH,str(y),gt), prediction)

	mground_truth = cv2.imread(os.path.join(GT_PATH,str(y),'maps', gt),cv2.IMREAD_GRAYSCALE)
	fground_truth = cv2.imread(os.path.join(GT_PATH,str(y),'fixation', gt),cv2.IMREAD_GRAYSCALE)
	other_map = construct_other()

	mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)
	fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
	other_map = cv2.resize(other_map, (0,0), fx=0.5, fy=0.5)
	saliency_map = cv2.resize(prediction, (0,0), fx=0.5, fy=0.5)


	mground_truth = mground_truth.astype(np.float32)
	fground_truth = fground_truth.astype(np.float32)
	other_map = other_map.astype(np.float32)
	saliency_map = saliency_map.astype(np.float32)

	# Calculate metrics
	AUC_judd = AUC_Judd(saliency_map, fground_truth)
	sAUC = AUC_shuffled(saliency_map, fground_truth, other_map)
	nss = NSS(saliency_map, fground_truth)
	cc = CC(saliency_map, mground_truth)
	sim = SIM(saliency_map, mground_truth)
	return ( AUC_judd, sAUC, nss, cc, sim)


assert (MODEL is not None), "Introduce model in args [--model]"
# create output file
if SAVE:
	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

# init model with pre-trained weights
if DEPTH and COORD:
	model = create_model(6)
elif DEPTH:
	model = create_model(4)
elif COORD:
	model = create_model(5)
else:
	model = create_model(3)
model = add_bn(model)
model.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS)['state_dict'])
model.eval()

# if GPU is enabled
if USE_GPU:
	model.cuda()

start = datetime.datetime.now().replace(microsecond=0)
for y in range(601,701):
	if SAVE:
		if not os.path.exists(os.path.join(OUTPUT_PATH,str(y))):
			os.makedirs(os.path.join(OUTPUT_PATH,str(y)))

	gt_path = os.path.join(GT_PATH, str(y))
	gt_files = os.listdir(os.path.join(gt_path,'maps'))
	gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]))

	predictions = inference(y)

	gt_prediction = zip(gt_files_sorted, predictions)
	from joblib import Parallel, delayed

	metric_list = Parallel(n_jobs=8)(delayed(inner_worker)(n, pack, y) for n, pack in enumerate(gt_prediction))
	aucj_mean = np.mean([x[0] for x in metric_list])
	aucs_mean = np.mean([x[1] for x in metric_list])
	nss_mean = np.mean([x[2] for x in metric_list])
	cc_mean = np.mean([x[3] for x in metric_list])
	sim_mean = np.mean([x[4] for x in metric_list])

	print("For video number {} the metrics are:".format(y))
	print("AUC-JUDD is {}".format(aucj_mean))
	print("AUC-SHUFFLED is {}".format(aucs_mean))
	print("NSS is {}".format(nss_mean))
	print("CC is {}".format(cc_mean))
	print("SIM is {}".format(sim_mean))
	print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
	print("==============================")
	message = 'For video number {} the metrics are:\nAUC-JUDD is {}\nAUC-SHUFFLED is {}\nNSS is {}\nCC is {}\nSIM is {}\nTime elapsed so far: {}\n=============================='.format(y,aucj_mean,aucs_mean,nss_mean,cc_mean,sim_mean,datetime.datetime.now().replace(microsecond=0)-start)
	send(message)
	final_metric_list.append(( aucj_mean,
						aucs_mean,
						nss_mean,
						cc_mean,
						sim_mean ))

Aucj = np.mean([y[0] for y in final_metric_list])
Aucs = np.mean([y[1] for y in final_metric_list])
Nss = np.mean([y[2] for y in final_metric_list])
Cc = np.mean([y[3] for y in final_metric_list])
Sim = np.mean([y[4] for y in final_metric_list])

print("Final average of metrics is:")
print("AUC-JUDD is {}".format(Aucj))
print("AUC-SHUFFLED is {}".format(Aucs))
print("NSS is {}".format(Nss))
print("CC is {}".format(Cc))
print("SIM is {}".format(Sim))
finalMessage = "Final average of metrics is:\nAUC-JUDD is {}\nAUC-SHUFFLED is {}\nNSS is {}\nCC is {}\nSIM is {}".format(Aucj,Aucs,Nss,Cc,Sim)
send(finalMessage)
print("TOTAL TIME: ", time.time()-start_total)

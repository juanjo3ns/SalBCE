from metrics_functions import AUC_Judd, AUC_shuffled, CC, NSS, SIM
import cv2
import os
import random
import numpy as np
import datetime
import time
#update status to my telegram account
import requests
import json
from IPython import embed
token = '658824281:AAETui7gl4muFLRsod1j2cGnuIDox_hj6hY'
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'
def send(message):
	 r = requests.post(
		 url=url,
		 data={'chat_id': 1458951, 'text': message}
		 #files = {'media': open('0087.jpg', 'rb')}
	 ).json()
#########
gt_directory = "/home/dataset/DHF1K/dhf1k_gt"
sm_directory = "/home/saliency_maps/salgan_dhf1k_adamoptim"
final_metric_list = []

total_start = time.time()

def construct_other():
	other_map = np.zeros((360,640))
	for i in range(0,50):
		random_video = random.randint(601, len(os.listdir(gt_directory)))
		other_map += cv2.imread(os.path.join(gt_directory, str(random_video),'fixation', '{:04d}.png'.format(random.randint(1, len(os.listdir(os.path.join(gt_directory, str(random_video),'fixation')))))),cv2.IMREAD_GRAYSCALE)
		other_map = np.clip(other_map,0,255)
	return other_map

def inner_worker(i, packed, gt_path, sm_path):

	gt, sm = packed

	mground_truth = cv2.imread(os.path.join(gt_path,'maps', gt),cv2.IMREAD_GRAYSCALE)
	fground_truth = cv2.imread(os.path.join(gt_path,'fixation', gt),cv2.IMREAD_GRAYSCALE)
	other_map = construct_other()
	saliency_map = cv2.imread(os.path.join(sm_path, sm),cv2.IMREAD_GRAYSCALE)

	mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)
	fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
	other_map = cv2.resize(other_map, (0,0), fx=0.5, fy=0.5)
	saliency_map = cv2.resize(saliency_map, (0,0), fx=0.5, fy=0.5)

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
	return ( AUC_judd, sAUC, nss, cc, sim )

# send('SALGAN METRICS')
start = datetime.datetime.now().replace(microsecond=0)
for i in range(601,701):

	gt_path = os.path.join(gt_directory, str(i))
	sm_path = os.path.join(sm_directory, str(i))

	gt_files = os.listdir(os.path.join(gt_path,'maps'))
	sm_files = os.listdir(sm_path)

	gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
	sm_files_sorted = sorted(sm_files, key = lambda x: int(x.split(".")[0]) )
	pack = zip(gt_files_sorted, sm_files_sorted)
	print("Files related to video {} sorted.".format(i))

	from joblib import Parallel, delayed
	metric_list = Parallel(n_jobs=1)(delayed(inner_worker)(n, packed, gt_path, sm_path) for n, packed in enumerate(pack))
	aucj_mean = np.mean([x[0] for x in metric_list])
	aucs_mean = np.mean([x[1] for x in metric_list])
	nss_mean = np.mean([x[2] for x in metric_list])
	cc_mean = np.mean([x[3] for x in metric_list])
	sim_mean = np.mean([x[4] for x in metric_list])

	print("For video number {} the metrics are:".format(i))
	print("AUC-JUDD is {}".format(aucj_mean))
	print("AUC-SHUFFLED is {}".format(aucs_mean))
	print("NSS is {}".format(nss_mean))
	print("CC is {}".format(cc_mean))
	print("SIM is {}".format(sim_mean))
	print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
	print("==============================")
	message = 'For video number {} the metrics are:\nAUC-JUDD is {}\nAUC-SHUFFLED is {}\nNSS is {}\nCC is {}\nSIM is {}\nTime elapsed so far: {}\n=============================='.format(i,aucj_mean,aucs_mean,nss_mean,cc_mean,sim_mean,datetime.datetime.now().replace(microsecond=0)-start)
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
print("COMPUTE METRICS TIME: ", time.time()-total_start)

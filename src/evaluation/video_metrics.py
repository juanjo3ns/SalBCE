from pprint import pprint
import os
import sys
from IPython import embed
import matplotlib.pylab as plt
import seaborn as sns
'''
videos = {
	601: {
		AUCJ: [value,value,value, ... ],
		AUCS: [value,value,value, ... ],
		NSS: [value,value,value, ... ],
		...
	},
	602: {
		AUCJ: [value,value,value, ... ],
		AUCS: [value,value,value, ... ],
		NSS: [value,value,value, ... ],
		...
	},
	603: {
		AUCJ: [value,value,value, ... ],
		AUCS: [value,value,value, ... ],
		NSS: [value,value,value, ... ],
		...
	},

	...
}

'''

def video_metrics():
	i=0
	videos = {}
	loss = {}

	valtrain = ['val', 'train']
	for v in valtrain:
		loss[v] =[]
	mm = ['AUC-J', 'AUC-S', 'NSS', 'CC', 'SIM']
	for num in range(601,701):
		videos[str(num)] = {}
	with open('7experiments.txt') as file:
		for line in file:
			line = line.strip()
			if 'For video' in line:
				i=0
				num = line.split(' ')[3]
			for x in mm:
				if x in line and i==0:
					if x not in videos[num]:
						videos[num][x]= []
					videos[num][x].append(float(line.split(' ')[2]))
			if 'average' in line or i==1:
				i = 1
			if 'val' in line or 'train' in line:
				vec = line.split(' ')
				loss[vec[0]].append(round(float(vec[4]),4))
	file.close()
	return videos
# #compute mean value of all metrics
# metrics = {}
# i=0
# current_exp=0
# for exp in range(len(loss['val'])):
# 	for m in mm:
# 		acc = 0
# 		for v in videos:
# 			acc += videos[v][m][exp]
# 		if current_exp not in metrics:
# 			metrics[current_exp]={}
# 			for x in range(0,3):
# 				metrics[current_exp][x] = {}
# 		metrics[current_exp][exp%3][m] = round(acc/100,3)
# 	i+=1
# 	if i%3==0:
# 		current_exp += 1
#
#
# #print results
# print("Number of experiments done: ", len(loss['train'])/3)
# print("Total epochs: ", len(loss['val']))
# pprint(metrics)
#
# maxperform = {}
#
# for m in mm:
# 	max = 0
# 	for exp in metrics:
# 		for epoch in range(0,3):
# 			if metrics[exp][epoch][m] > max:
# 				max = metrics[exp][epoch][m]
# 				maxperform[m] = (exp, epoch)
# 	print("Better {} corresponds to:\n\tExperiment: {}\n\tEpoch: {}".format(m,maxperform[m][0],maxperform[m][1]))
#
# # embed()
#
# # plt.show()
# # for v in valtrain:
# # 	plt.plot(loss[v])
# # plt.show()
#
#
# for n in [0,3,6,9,12]:
# 	fig, ax = plt.subplots(nrows=2, ncols=3)
# 	for num in videos:
# 		ax[0,0].plot(videos[num]['AUC-J'][n:n+3])
# 		ax[0,0].set_title('AUC-J')
# 	for num in videos:
# 		ax[0,1].plot(videos[num]['AUC-S'][n:n+3])
# 		ax[0,1].set_title('AUC-S')
# 	for num in videos:
# 		ax[0,2].plot(videos[num]['NSS'][n:n+3])
# 		ax[0,2].set_title('NSS')
# 	for num in videos:
# 		ax[1,0].plot(videos[num]['CC'][n:n+3])
# 		ax[1,0].set_title('CC')
# 	for num in videos:
# 		ax[1,1].plot(videos[num]['SIM'][n:n+3])
# 		ax[1,1].set_title('SIM')
# 	for l in loss:
# 		ax[1,2].plot(loss[l][n:n+3])
# 		ax[1,2].set_title(l)
# 	plt.show()
# # print(videos)
# file.close()

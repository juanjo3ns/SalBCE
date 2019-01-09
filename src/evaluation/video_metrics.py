from pprint import pprint
import os
import sys
from IPython import embed
import matplotlib.pylab as plt
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
i=0
videos = {}
loss = {}

valtrain = ['val', 'train']
for v in valtrain:
	loss[v] =[]
mm = ['AUC-J', 'AUC-S', 'NSS', 'CC', 'SIM']
for num in range(601,701):
	videos[str(num)] = {}
with open('alluntil9january.txt') as file:
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

#compute mean value of all metrics
metrics = {}
i=0
current_exp=0
for exp in range(len(loss['val'])):
	for m in mm:
		acc = 0
		for v in videos:
			acc += videos[v][m][exp]
		if current_exp not in metrics:
			metrics[current_exp]={}
			for x in range(0,4):
				metrics[current_exp][x] = {}
		metrics[current_exp][exp%4][m] = round(acc/100,3)
	if i%3==0:
		current_exp += 1
	i+=1

#print results
print("Number of experiments done: ", len(loss['train'])/4)
print("Total epochs: ", len(loss['train']))
pprint(metrics)
# for m in mm:
# 	max = max()
# 	epoch = videos[m][] - 1
# 	print("Better {} corresponds to:\n\tExperiment: {}\n\tEpoch: {}".format(m,exp,epoch))

# embed()

# plt.show()
# for v in valtrain:
# 	plt.plot(loss[v])
# plt.show()



fig, ax = plt.subplots(nrows=2, ncols=3)
for num in videos:
	ax[0,0].plot(videos[num]['AUC-J'])
	ax[0,0].set_title('AUC-J')
for num in videos:
	ax[0,1].plot(videos[num]['AUC-S'])
	ax[0,1].set_title('AUC-S')
for num in videos:
	ax[0,2].plot(videos[num]['NSS'])
	ax[0,2].set_title('NSS')
for num in videos:
	ax[1,0].plot(videos[num]['CC'])
	ax[1,0].set_title('CC')
for num in videos:
	ax[1,1].plot(videos[num]['SIM'])
	ax[1,1].set_title('SIM')
for l in loss:
	ax[1,2].plot(loss[l])
	ax[1,2].set_title(l)
# plt.show()
# print(videos)
file.close()

import os

from dataloader.datasetDHF1K import DHF1K
from torch.utils.data import DataLoader
from utils.salgan_utils import save_model, get_lr_optimizer
from utils.sendTelegram import send
from utils.salgan_generator import create_model, add_bn
from evaluation.fast_evaluation import compute_metrics

import numpy as np

import torch
from torch.nn import AvgPool2d
from torch.nn.modules.loss import BCELoss
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


from IPython import embed
from tensorboard_logger import configure, log_value, log_histogram


TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def add_layer_weights(vgg_weights):
	# Mean of RGB weights of first layer with size [64,1,3,3]
	layer1 = vgg_weights['0.weight']
	mean_rgb = layer1.mean(dim=1,keepdim=True)
	vgg_weights['0.weight'] = torch.cat([layer1.cuda(),mean_rgb.cuda()],1)
	# We could do it easily accessing to the weights trought model[0].weight and change dimension 1, but as we
	# already have the 4th channel we'd be doing the mean of all of the channels, inicializing it in the wrong way.
	return vgg_weights

def train_eval(mode, model, optimizer, dataloader):
	if mode == TRAIN:
		N = len(ds_train)/batch_size
		model.train()
	else:
		N = len(ds_validate)/batch_size
		model.eval()

	total_loss = []
	#iterate epoch...
	for i, X in enumerate(dataloader[mode]):
		inputs = X[0].cuda()
		# noramlize saliency maps values between [0,1]
		gt_maps = X[1].cuda()/255
		predictions = model.forward(inputs).squeeze()

		# reduce size for loss
		reduce_size = AvgPool2d((4,4))
		pred_ = reduce_size(predictions)
		gt_maps_ = reduce_size(gt_maps)

		pred_ = pred_.view(pred_.size()[0], -1)
		gt_maps_ = gt_maps_.view(gt_maps_.size()[0], -1)

		loss = bce_loss(pred_, gt_maps_)

		# make actual step update
		if mode==TRAIN:
			# compute gradients
			loss.backward()
			# step optimizer
			optimizer.step()
			# reset grads for next step
			optimizer.zero_grad()


		print("\t{}/{} loss:{}".format(i, int(N), loss.item()), end="\r")
		total_loss.append(loss.item())

	total_loss=np.mean(total_loss)
	return total_loss



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_out", default='sal_dhf1k_adamdepthcoordaugm2_frombestsaldepth',
				type=str,
				help="""set output path for the trained model""")
	parser.add_argument("--batch_size", default=32,
				type=int,
				help="""Set batch size""")
	parser.add_argument("--n_epochs", default=5, type=int,
				help="""Set total number of epochs""")
	parser.add_argument("--depth", default=False, type=bool,
				help="""Enable 4th channel with depth""")
	parser.add_argument("--augment", default=False, type=bool,
				help="""Enable data augmentation""")
	parser.add_argument("--coord", default=False, type=bool,
				help="""Enable coordconv""")
	parser.add_argument("--lr", type=float, default=0.00001,
				help="""Learning rate for training""")
	parser.add_argument("--patience", type=int, default=2,
				help="""Patience for learning rate scheduler (default 10)""")
	args = parser.parse_args()


	# set output path ==========================================================
	path_out = '../trained_models/batch32/' + args.path_out

	if not os.path.exists(path_out):
		# create output path
		os.makedirs(path_out)

		# create output for models
		path_models = os.path.join(path_out, 'models')
		if not os.path.exists(path_models):
			os.makedirs(path_models)

	# tensorboard
	configure("{}".format(path_out), flush_secs=5)

	# data =====================================================================
	batch_size = args.batch_size
	n_epochs = args.n_epochs
	lr = args.lr
	DEPTH = args.depth
	AUGMENT = args.augment
	COORD = args.coord
	# Datasets for DHF1K
	ds_train = DHF1K(mode=TRAIN, transformation=True, depth=DEPTH, d_augm=AUGMENT, coord=COORD)
	ds_validate = DHF1K(mode=VAL, transformation=True, depth=DEPTH, d_augm=AUGMENT, coord=COORD)

	# Dataloaders
	dataloader = {
		TRAIN: DataLoader(ds_train, batch_size=batch_size,
								shuffle=True, num_workers=2),
		VAL: DataLoader(ds_validate, batch_size=batch_size,
								shuffle=False, num_workers=2)
	}



	torch.cuda.set_device(1)
	# model ====================================================================
	print("Init model...")
	# init model with pre-trained weights
	vgg_weights = torch.load('../trained_models/salgan_baseline.pt')['state_dict']
	if DEPTH and COORD:
		model = create_model(6)
		for i in range(0,3):
			vgg_weights = add_layer_weights(vgg_weights)
	elif DEPTH:
		model = create_model(4)
		add_layer_weights(vgg_weights)
	elif COORD:
		model = create_model(5)
		for i in range(0,2):
			vgg_weights = add_layer_weights(vgg_weights)
	else: model = create_model(3)
	model.load_state_dict(vgg_weights)

	# Add batch normalization to current model
	model = add_bn(model)

	model.train()
	model.cuda()
	cudnn.benchmark = True

	# NOT WORKING UNMOUNTED DISK
	# If we have the two GPU's available we are going to use both
	# if torch.cuda.device_count() > 1:
	# 	print("Using ", torch.cuda.device_count(), "GPUs!")
	# 	model = torch.nn.DataParallel(model)




	# loss =====================================================================
	print("BCE criterium...")
	bce_loss = BCELoss()
	trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Trainable parameters: ", trainable_parameters)
	send("Trainable parameters: " + str(trainable_parameters))
	send("Experiment: " + args.path_out)

	# select only decoder parameters, keep vgg16 with pretrained weights
	decoder_parameters = []
	base_params = []
	for i, (a, p) in enumerate(model.named_parameters()):
		if i>25:
			print(i, a, p.shape)
			decoder_parameters.append(p)
		else: base_params.append(p)

	# ADAM OPTIMIZER
	optimizer = Adam(model.parameters(),
					lr = lr,
					weight_decay=0.000001)

	# STOCHASTIC GRADIENT DESCENT
	# optimizer = SGD(model.parameters(),
	# 				lr = 0.00001,
	# 				momentum=0.9,
	# 				weight_decay=0.00001,
	# 				nesterov=True)

	# set learning rate scheduler
	# ReduceLROnPlateau(
		# optimizer,
		# mode (str) 'min':lr es reduira quan la metrica no es redueixi mes,  'max' al contrari,
		# factor (float) factor de reduccio de la lr,
		# patience (int) num epochs sense millora a partir dels quals es redueix lr,
		# verbose (bool),
	# )
	# scheduler = ReduceLROnPlateau(optimizer,
	# 							'min',
	# 							patience=args.patience,
	# 							verbose=True)
	scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

	best_loss=9999999

	# main loop training =======================================================
	for id_epoch in range(n_epochs):
		for mode in [VAL, TRAIN]:
			# select dataloader
			data_iterator = dataloader[mode]

			# saliency metrics
			if mode ==VAL:
				print("Evaluating metrics....")
				# only do 100 images from validation
				metrics = compute_metrics(model, 100, DEPTH, COORD)

				# log metric values
				for metric in metrics.keys():
					log_value("Metrics/{}".format(metric),
								metrics[metric], id_epoch)

			# get epoch loss
			print("--> {} epoch {}".format(mode, id_epoch))


			epoch_loss = train_eval(mode, model, optimizer, dataloader)

			lr = list(get_lr_optimizer(optimizer))[0]
			print("-----------")
			print("Done! {} epoch {} loss {} lr {}".format(mode, id_epoch, epoch_loss, lr))
			send("{} epoch {}/{} loss {}".format(mode, id_epoch, n_epochs, epoch_loss))
			print("\n")

			# record loss
			log_value("loss/{}".format(mode), epoch_loss, id_epoch)
			log_value("lr/{}".format(mode), lr, id_epoch)
			# for v in model.state_dict():
			# 	log_histogram("Layer {}".format(v), model.state_dict()[v], id_epoch)
			# if (id_epoch%10)==0:
			# 	save_model(model, optimizer, id_epoch, path_out, name_model='{:03d}'.format(id_epoch))
			# store model if val loss improves
			if mode==VAL:
				if best_loss > epoch_loss:
					# update loss
					best_loss = epoch_loss

					save_model(model, optimizer, id_epoch, path_out, name_model='best')
				# scheduler.step(epoch_loss)
				scheduler.step()

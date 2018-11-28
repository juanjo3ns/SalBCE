import os

from dataloader.datasetSALICON_adv import SALICON
from torch.utils.data import DataLoader
from utils.salgan_utils import save_model, get_lr_optimizer
from utils.sendTelegram import send
from utils.salgan_generator import create_model
from utils.salgan_discriminator import Discriminator
import numpy as np

import torch
from torch.nn import AvgPool2d
from torch.nn.modules.loss import BCELoss
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau


from IPython import embed
from tensorboard_logger import configure, log_value, log_histogram
import requests
import json
token = '658824281:AAETui7gl4muFLRsod1j2cGnuIDox_hj6hY'
url = 'https://api.telegram.org/bot' + token+ '/sendMessage'
def send(message):
	r = requests.post(
		url=url,
		data={'chat_id': 1458951, 'text': message}
		#files = {'media': open('0087.jpg', 'rb')}
	).json()

TRAIN = 'train'
VAL = 'val'
TEST = 'test'


def train_eval(mode, modelG, modelD, optimizerG,optimizerD, dataloader):
	if mode == TRAIN:
		N = len(ds_train)/batch_size
		modelG.train()
		modelD.train()
	else:
		N = len(ds_validate)/batch_size
		modelG.eval()
		modelD.eval()

	generator_loss = []
	discriminator_loss = []
	#iterate epoch...
	# For each batch we alternate between generator training and adversarial training
	for (i,X), discr in zip(enumerate(dataloader[mode]), [True,False]*(int(len(dataloader[mode])/2)+1)):

	# for i, X in enumerate(dataloader[mode]):

		if discr:
			inputs = X[0].cuda()
			gt_maps = X[1].cuda()/255
			predictions = modelG.forward(inputs)

			# Torch shape = [batch, channels, height, width]
			# We want four channels with image (RGB) + predictor or saliency map because we use ConditionalGAN
			RGBP = torch.cat([inputs,predictions], 1)
			RGBS = torch.cat([inputs,gt_maps.unsqueeze(1)], 1)
			pred_pred = modelD.forward(RGBP)
			pred_gt = modelD.forward(RGBPS)

			# Loss during the training of the discriminator
			loss_pred = bce_loss(pred_pred, torch.zeros([10,1]).cuda())
			loss_gt = bce_loss(pred_gt, torch.ones([10,1]).cuda())
			discr_loss = loss_pred + loss_gt
			# print("\Discriminator {}/{} loss:{}".format(i, int(N), discr_loss.item()))
			discriminator_loss.append(discr_loss.item())
			# make actual step update
			if mode==TRAIN:
				# reset grads for next step
				optimizerD.zero_grad()
				# compute gradients
				# loss_pred.backward(keep_graph=True)
				# loss_gt.backward()
				discr_loss.backward()
				# step optimizer
				optimizerD.step()

			print("\t{}: {}/{} loss:{}".format('Discriminator',i, int(N), discr_loss.item()))
		else:
			inputs = X[0].cuda()
			gt_maps = X[1].cuda()/255
			predictions = modelG.forward(inputs)
			RGBP = torch.cat([inputs,predictions], 1)
			discr_pred = modelD.forward(RGBP)
			# reduce size for loss
			reduce_size = AvgPool2d((4,4))
			pred_ = reduce_size(predictions.squeeze())
			gt_maps_ = reduce_size(gt_maps)

			pred_ = pred_.view(pred_.size()[0], -1)
			gt_maps_ = gt_maps_.view(gt_maps_.size()[0], -1)

			# alpha multiplying??
			loss = 0.005*bce_loss(pred_, gt_maps_) + bce_loss(discr_pred, torch.ones([10,1]).cuda()) #????????????????
			print("\tGenerator {}/{} loss:{}\033[F".format(i, int(N), loss.item()), end="\r")
			generator_loss.append(loss.item())
			# make actual step update
			if mode==TRAIN:
				# reset grads for next step
				optimizerG.zero_grad()
				# compute gradients
				loss.backward()
				# step optimizer
				optimizerG.step()



		# print("\t{}/{} loss:{}".format(i, int(N), loss.item()))
		# total_loss.append(loss.item())

	total_discriminator_loss=np.mean(discriminator_loss)
	total_generator_loss=np.mean(generator_loss)

	return total_discriminator_loss, total_generator_loss



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_out", default='../trained_models/salgan_salicon_adversarial2',
				type=str,
				help="""set output path for the trained model""")
	parser.add_argument("--batch_size", default=10,
				type=int,
				help="""Set batch size""")
	parser.add_argument("--n_epochs", default=100, type=int,
				help="""Set total number of epochs""")
	parser.add_argument("--lr", type=float, default=0.0003,
				help="""Learning rate for training""")
	parser.add_argument("--patience", type=int, default=5,
				help="""Patience for learning rate scheduler (default 10)""")
	args = parser.parse_args()


	# set output path ==========================================================
	path_out = args.path_out

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

	# Datasets for DHF1K
	ds_train = SALICON(mode=TRAIN)
	ds_validate = SALICON(mode=VAL)

	# Dataloaders
	dataloader = {
		TRAIN: DataLoader(ds_train, batch_size=batch_size,
								shuffle=True, num_workers=4),
		VAL: DataLoader(ds_validate, batch_size=batch_size,
								shuffle=False, num_workers=4)
	}




	# model ====================================================================
	print("Init model...")
	# Init generator model with pretrained weights from SALICON's
	modelG = create_model()
	modelG.load_state_dict(torch.load('../trained_models/salgan_salicon_3epochs/models/best.pt')['state_dict'])
	modelG.train()
	modelG.cuda()
	embed()
	# Init discriminator model with weights ????????
	modelD = Discriminator()
	#modelD.load_state_dict(torch.load(????)['state_dict'])
	modelD.train()
	modelD.cuda()

	cudnn.benchmark = True

	# loss =====================================================================
	print("BCE criterium...")
	bce_loss = BCELoss()

	# select only decoder parameters, keep vgg16 with pretrained weights
	decoder_parameters = []
	for i, (a, p) in enumerate(modelG.named_parameters()):
		if i>25:
			print(i, a, p.shape)
			decoder_parameters.append(p)

	# Generator optimizer
	optimizerG = SGD(decoder_parameters,
					lr = args.lr,
					momentum=0.9,
					weight_decay=0.00001)

	# Discriminator optimizer
	optimizerD = SGD(decoder_parameters,
					lr = args.lr,
					weight_decay=0.00001)

	# LR generator scheduler
	schedulerG = ReduceLROnPlateau(optimizerG,
								'min',
								patience=args.patience,
								verbose=True)
	# LR discriminator scheduler
	schedulerD = ReduceLROnPlateau(optimizerD,
								'min',
								patience=args.patience,
								verbose=True)

	best_lossG=9999999
	best_lossD=9999999

	# main loop training =======================================================
	for id_epoch in range(n_epochs):
		for mode in [VAL, TRAIN]:
			# select dataloader
			data_iterator = dataloader[mode]

			# compute saliency metrics
			# (not sure if this metrics are right thought)
			# if mode ==VAL:
			#	 print("Evaluating metrics....")
			#	 # only do 100 images from validation (too slow if not)
			#	 metrics = get_saliency_metrics(data_iterator, model, N=100)
			#
			#	 # log metric values
			#	 for metric in metrics.keys():
			#		 log_value("Metrics/{}".format(metric),
			#					 metrics[metric], id_epoch)

			# get epoch loss
			print("--> {} epoch {}".format(mode, id_epoch))


			epoch_lossG, epoch_lossD = train_eval(mode, modelG, modelD, optimizerG, optimizerD, dataloader)

			lrG = list(get_lr_optimizer(optimizerG))[0]
			lrD = list(get_lr_optimizer(optimizerD))[0]
			print("-----------")
			print("\033[FDone! {} epoch {} \n\tGenerator loss: {}\n\tDiscrimintator loss {}".format(mode, id_epoch, epoch_lossG, epoch_lossD))
			send("Done! {} epoch {} \n\tGenerator loss: {}\n\tDiscrimintator loss {}".format(mode, id_epoch, epoch_lossG, epoch_lossD))
			print("\n")

			# record loss
			log_value("lossG/{}".format(mode), epoch_lossG, id_epoch)
			log_value("lrG/{}".format(mode), lrG, id_epoch)
			log_value("lossD/{}".format(mode), epoch_lossD, id_epoch)
			log_value("lrD/{}".format(mode), lrD, id_epoch)
			# for v in modelG.state_dict():
			# 	log_histogram("Layer {}".format(v), modelG.state_dict()[v], id_epoch)

			# save_model(model, optimizer, id_epoch, path_out, name_model='{:03d}'.format(id_epoch))
			# store model if val loss improves
			if mode==VAL:
				if best_lossG > epoch_lossG:
					# update loss
					best_lossG = epoch_lossG

					save_model(modelG, optimizerG, id_epoch, path_out, name_model='bestG')
				if best_lossD > epoch_lossD:
					# update loss
					best_lossD = epoch_lossD

					save_model(modelD, optimizerD, id_epoch, path_out, name_model='bestD')
				schedulerG.step(epoch_lossG)
				schedulerD.step(epoch_lossD)

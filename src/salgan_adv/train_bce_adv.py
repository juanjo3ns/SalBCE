import os

from dataloader.datasetSALICON import SALICON
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


def train_eval(mode, modelG, modelD, optimizerG, dataloader):
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
	for (i,X), discr in zip(enumerate(dataloader[mode]), [True,False]*(len(dataloader[mode])/2)):

	# for i, X in enumerate(dataloader[mode]):
		embed()
		if discr:
			# add to inputs one more layer [batch, height, width, channels+predicted/groundtruth]
			inputs = X[0].cuda()
			gt_maps = X[1].cuda()/255
			predictions = modelG.forward(inputs).squeeze()


			# reduce size for loss
			reduce_size = AvgPool2d((4,4))
			predictions = reduce_size(predictions)
			gt_maps = reduce_size(gt_maps)

			predictions = predictions.view(pred_.size()[0], -1)
			gt_maps = gt_maps.view(gt_maps_.size()[0], -1)


			pred_pred = modelD.forward(predictions).squeeze()
			pred_gt = modelD.forward(gt_maps).squeeze()
			# Loss during the training of the discriminator
			loss_pred = BCELoss(pred_pred, 1)???
			loss_gt = BCELoss(pred_gt, 0)???
			discr_loss = loss_pred + loss_gt
			print("\Discriminator {}/{} loss:{}".format(i, int(N), discr_loss.item()), end="\r")
			discriminator_loss.append(discr_loss.item())
			# make actual step update
			if mode==TRAIN:
				# reset grads for next step
				optimizerD.zero_grad()
				# compute gradients
				loss_pred.backward()
				loss_gt.backward()
				# step optimizer
				optimizerD.step()

		else:
			inputs = X[0].cuda()
			gt_maps = X[1].cuda()/255
			predictions = modelG.forward(inputs).squeeze()

			# reduce size for loss
			reduce_size = AvgPool2d((4,4))
			pred_ = reduce_size(predictions)
			gt_maps_ = reduce_size(gt_maps)

			pred_ = pred_.view(pred_.size()[0], -1)
			gt_maps_ = gt_maps_.view(gt_maps_.size()[0], -1)

			loss = alpha*BCELoss(pred_, gt_maps_) + BCELoss(pred_gt, 1)???
			print("\tGenerator {}/{} loss:{}".format(i, int(N), loss.item()), end="\r")
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
	parser.add_argument("--path_out", default='trained_models/salgan_salicon_adversarial',
				type=str,
				help="""set output path for the trained model""")
	parser.add_argument("--batch_size", default=32,
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
	ds_train = SALICON(mode=TRAIN, N=32*5)
	ds_validate = SALICON(mode=VAL, N=32*3)

	# Dataloaders
	dataloader = {
		TRAIN: DataLoader(ds_train, batch_size=batch_size,
								shuffle=True, num_workers=2),
		VAL: DataLoader(ds_validate, batch_size=batch_size,
								shuffle=False, num_workers=2)
	}




	# model ====================================================================
	print("Init model...")
	# Init generator model with pretrained weights from SALICON's
	modelG = create_model()
	modelG.load_state_dict(torch.load('../trained_models/salgan_salicon_daugmfromscr3/models/best.pt')['state_dict'])
	modelG.train()
	modelG.cuda()
	# Init discriminator model with weights ????????

	modelD = Discriminator()
	modelD.load_state_dict(torch.load(????)['state_dict'])
	modelD.train()
	modelD.cuda()
	cudnn.benchmark = True

	# select only decoder parameters, keep vgg16 with pretrained weights
	decoder_parameters = []
	for i, (a, p) in enumerate(modelG.named_parameters()):
		if i>25:
			print(i, a, p.shape)
			decoder_parameters.append(p)

	optimizerG = SGD(decoder_parameters,
					lr = args.lr,
					momentum=0.9,
					weight_decay=0.00001)
	optimizerD = Adagrad(decoder_parameters,
					lr = args.lr,
					weight_decay=0.00001)

	# set learning rate scheduler
	# ReduceLROnPlateau(
		# optimizer,
		# mode (str) 'min':lr es reduira quan la metrica no es redueixi mes,  'max' al contrari,
		# factor (float) factor de reduccio de la lr,
		# patience (int) num epochs sense millora a partir dels quals es redueix lr,
		# verbose (bool),
	# )
	scheduler = ReduceLROnPlateau(optimizerG,
								'min',
								patience=args.patience,
								verbose=True)

	best_loss=9999999

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

			lr = list(get_lr_optimizer(optimizerG))[0]
			print("-----------")
			print("Done! {} epoch {} \n\tGenerator loss: {}\n\tDiscrimintator loss {}".format(mode, id_epoch, epoch_lossG, epoch_lossD))
			send("Done! {} epoch {} \n\tGenerator loss: {}\n\tDiscrimintator loss {}".format(mode, id_epoch, epoch_lossG, epoch_lossD))
			print("\n")

			# record loss
			log_value("loss/{}".format(mode), epoch_loss, id_epoch)
			log_value("lr/{}".format(mode), lr, id_epoch)
			for v in modelG.state_dict():
				log_histogram("Layer {}".format(v), modelG.state_dict()[v], id_epoch)

			# save_model(model, optimizer, id_epoch, path_out, name_model='{:03d}'.format(id_epoch))
			# store model if val loss improves
			if mode==VAL:
				if best_loss > epoch_loss:
					# update loss
					best_loss = epoch_loss

					save_model(modelG, optimizerG, id_epoch, path_out, name_model='best')
				scheduler.step(epoch_loss)

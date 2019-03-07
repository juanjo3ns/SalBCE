import os

from dataloader.datasetLEDOV import LEDOV
from torch.utils.data import DataLoader
from utils.salgan_utils import save_model, get_lr_optimizer
from utils.sendTelegram import send
from utils.salgan_generator import create_model

import numpy as np

import torch
from torch.nn import AvgPool2d
from torch.nn.modules.loss import BCELoss
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


from IPython import embed
from tensorboard_logger import configure, log_value, log_histogram


TRAIN = 'train'
VAL = 'val'
TEST = 'test'

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
		if not predictions.shape == torch.Size([9, 192, 256]):
			embed()
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
	parser.add_argument("--path_out", default='../trained_models/salgan_ledov_from27',
				type=str,
				help="""set output path for the trained model""")
	parser.add_argument("--batch_size", default=9,
				type=int,
				help="""Set batch size""")
	parser.add_argument("--n_epochs", default=3, type=int,
				help="""Set total number of epochs""")
	parser.add_argument("--lr", type=float, default=0.001,
				help="""Learning rate for training""")
	parser.add_argument("--patience", type=int, default=3,
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

	# Datasets for LEDOV
	ds_train = LEDOV(mode=TRAIN,transformation=True)
	ds_validate = LEDOV(mode=VAL,transformation=True)

	# Dataloaders
	dataloader = {
		TRAIN: DataLoader(ds_train, batch_size=batch_size,
								shuffle=True, num_workers=2),
		VAL: DataLoader(ds_validate, batch_size=batch_size,
								shuffle=False, num_workers=2)
	}




	# model ====================================================================
	print("Init model...")
	# init model with pre-trained weights
	model = create_model()
	model.load_state_dict(torch.load('../trained_models/salgan_salicon_27epoch/models/best.pt')['state_dict'])
	model.train()
	model.cuda()
	#allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
	cudnn.benchmark = True

	# loss =====================================================================
	print("BCE criterium...")
	bce_loss = BCELoss()

	# select only decoder parameters, keep vgg16 with pretrained weights
	decoder_parameters = []
	for i, (a, p) in enumerate(model.named_parameters()):
		if i>25:
			print(i, a, p.shape)
			decoder_parameters.append(p)

	optimizer = SGD(decoder_parameters,
					lr = args.lr,
					momentum=0.9,
					weight_decay=0.00001,
					nesterov=True)

	# set learning rate scheduler
	# ReduceLROnPlateau(
		# optimizer,
		# mode (str) 'min':lr es reduira quan la metrica no es redueixi mes,  'max' al contrari,
		# factor (float) factor de reduccio de la lr,
		# patience (int) num epochs sense millora a partir dels quals es redueix lr,
		# verbose (bool),
	# )
	scheduler = ReduceLROnPlateau(optimizer,
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
			# 	print("Evaluating metrics....")
			# 	# only do 100 images from validation (too slow if not)
			# 	metrics = get_saliency_metrics(data_iterator, model, N=100)

			# 	# log metric values
			# 	for metric in metrics.keys():
			# 		log_value("Metrics/{}".format(metric),
			# 					metrics[metric], id_epoch)

			# get epoch loss
			print("--> {} epoch {}".format(mode, id_epoch))

			epoch_loss = train_eval(mode, model, optimizer, dataloader)

			lr = list(get_lr_optimizer(optimizer))[0]
			print("-----------")
			print("Done! {} epoch {} loss {}".format(mode, id_epoch, epoch_loss))
			send("{} epoch {}/{} loss {}".format(mode, id_epoch, n_epochs, epoch_loss))
			print("\n")

			# record loss
			log_value("loss/{}".format(mode), epoch_loss, id_epoch)
			log_value("lr/{}".format(mode), lr, id_epoch)
			for v in model.state_dict():
				log_histogram("Layer {}".format(v), model.state_dict()[v], id_epoch)

			# save_model(model, optimizer, id_epoch, path_out, name_model='{:03d}'.format(id_epoch))
			# store model if val loss improves
			if mode==VAL:
				if best_loss > epoch_loss:
					# update loss
					best_loss = epoch_loss

					save_model(model, optimizer, id_epoch, path_out, name_model='best')
				scheduler.step(epoch_loss)

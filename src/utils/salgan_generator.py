import torch
from torchvision.models import vgg16
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn.modules import BatchNorm2d
from IPython import embed
import torch

def add_bn(model):
	seq = list(model)
	relu_positions = [(59,64),(57,64),(54,128),(52,128),(49,256),(47,256),(45,256),
	(42,512),(40,512),(38,512),(35,512),(33,512),(31,512),(29,512),(27,512),(25,512),
	(22,512),(20,512),(18,512),(15,256),(13,256),(11,256),(8,128),(6,128),(3,64),(1,64)]
	for pos, channels in relu_positions:
		seq.insert(pos,BatchNorm2d(channels))
	return torch.nn.Sequential(*seq)

def create_model(input_channels):
	# Create encoder based on VGG16 architecture
	# original_vgg16 = vgg16()
	#
	# # select only convolutional layers
	# encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])

	# new enconder
	encoder = [
		Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
		Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
		Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
		Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU()]

	# define decoder based on VGG16 (inverse order and Upsampling layers)
	decoder_list=[
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Upsample(scale_factor=2, mode='nearest'),

		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Upsample(scale_factor=2, mode='nearest'),

		Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Upsample(scale_factor=2, mode='nearest'),

		Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Upsample(scale_factor=2, mode='nearest'),

		Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		ReLU(),
		Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
		Sigmoid(),
	]
	encoder = torch.nn.Sequential(*encoder)
	decoder = torch.nn.Sequential(*decoder_list)

	# assamble the full architecture encoder-decoder
	model = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))

	return model

if __name__ == '__main__':
	model = create_model(4)
	embed()

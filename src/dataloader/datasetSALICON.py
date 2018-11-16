import os
import sys
import cv2

import numpy as np
from IPython import embed

import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# constants
PATH_SALICON = "/media/eva/WD8T/Datasets/SALICON/"


class Salicon(Dataset):
    def __init__(self, mode='train', transformation=False, return_path=False, N=None):
        global PATH_SALICON
        self.size = (192, 256)
        self.mean = [103.939, 116.779, 123.68]
        self.path_dataset = PATH_SALICON
        self.path_images = os.path.join(self.path_dataset, 'images')
        self.path_saliency = os.path.join(self.path_dataset, 'saliency')
        self.transformation = transformation
        self.return_path = return_path

        # get list images
        list_names = os.listdir( os.path.join(self.path_dataset, 'fixations', mode) )
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        # set path
        ima_name = self.list_names[index]+'.jpg'
        rgb_ima = os.path.join(self.path_images, ima_name)
        image = cv2.imread(rgb_ima)
        ima_name = self.list_names[index]+'.png'
        saliency = cv2.imread(os.path.join(self.path_saliency, ima_name), 0)

        # apply transformation
        if self.transformation is not None:
            # reshape
            image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
            saliency = cv2.resize(saliency, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

            # convert to foat
            image = image.astype(np.float32)
            saliency = saliency.astype(np.float32)

            # remove mean value
            image -= self.mean

            # convert to torch Tensor
            image = torch.FloatTensor(image)

            # swap channel dimensions
            image = image.permute(2,0,1)


        if self.return_path:
            return image, saliency, rgb_ima
        else:
            return image, saliency


if __name__ == '__main__':
    ds = Salicon(mode='train')

    print(len(ds))
    image, saliency = ds[0]

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(saliency)
    plt.show()
    embed()

import torch

import torch.nn.functional as F
from torch.nn import MaxPool2d, Linear, Module, Sigmoid, Tanh
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid

from IPython import embed


class Discriminator(Module):
    def __init__(self):

        super(Discriminator, self).__init__()
        self.conv1_1 = Conv2d(4, 3, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2_1 = Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3_1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.fc4 = Linear(49152, 100)
        self.fc5 = Linear(100, 2)
        self.fc6 = Linear(2, 1)

        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1(x))
        x = F.leaky_relu(self.conv1_2(x))
        x = self.pool(x)

        x = F.leaky_relu(self.conv2_1(x))
        x = F.leaky_relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.leaky_relu(self.conv3_1(x))
        x = F.leaky_relu(self.conv3_2(x))
        x = self.pool(x)

        # flatten conv
        x = x.view(x.size()[0], -1)
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.sigmoid(self.fc6(x))

        return x

def create_model():
    return Discriminator()

if __name__ == '__main__':
    X = torch.randn((1,3,196,256))

    model = Discriminator()
    embed()

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, use_gpu, input_size, hidden_size, kernel_size):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(in_channels = input_size + hidden_size, out_channels = 4 * hidden_size, kernel_size = kernel_size, padding = 1) #padding 1 to preserve HxW dimensions
        self.conv1x1 = nn.Conv2d(in_channels = hidden_size, out_channels = 1, kernel_size = 1)

    def forward(self, input_, prev_state=None):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]


        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )


        prev_hidden, prev_cell = prev_state


        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = f.sigmoid(in_gate)
        forget_gate = f.sigmoid(forget_gate)
        out_gate = f.sigmoid(out_gate)

        cell_gate = f.tanh(cell_gate)

        forget = (forget_gate * prev_cell)
        update = (in_gate * cell_gate)
        cell = forget + update
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]
        saliency_map = self.conv1x1(cell)

        return (hidden, cell), saliency_map



class Conv(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, use_gpu, input_size, filter_size, kernel_size):
        super(Conv,self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.conv = nn.Conv2d(in_channels = input_size, out_channels = filter_size, kernel_size = kernel_size, padding = 1) #padding 1 to preserve HxW dimensions
        self.conv1x1 = nn.Conv2d(in_channels = filter_size, out_channels = 1, kernel_size = 1)

    def forward(self, input_, prev_state=None):

        x = self.conv(input_)
        saliency_map = self.conv1x1(x)

        return saliency_map

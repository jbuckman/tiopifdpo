import math
from numpy import prod
from torch import nn
from torch.nn import functional as F

class FFNet(nn.Module):
    def __init__(self, in_size, layers, out_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(_in, _out)
            for _in, _out in zip([in_size] + list(layers), list(layers) + [out_size])
        ])
    def reset_parameters(self):
        for layer in self.layers: layer.reset_parameters()
    def forward(self, x):
        for layer in self.layers[:-1]: x = F.leaky_relu(layer(x))
        return self.layers[-1](x)

def conv2d_out_shape(in_shape, conv2d):
    C = conv2d.out_channels
    H = math.floor((in_shape[-2] + 2.*conv2d.padding[0] - conv2d.dilation[0] * (conv2d.kernel_size[0] - 1.) - 1.)/conv2d.stride[0] + 1.)
    W = math.floor((in_shape[-1] + 2.*conv2d.padding[1] - conv2d.dilation[1] * (conv2d.kernel_size[1] - 1.) - 1.)/conv2d.stride[1] + 1.)
    return [C, H, W]
cs = conv2d_out_shape

class QConvNet(nn.Module):
    def __init__(self, state_shape, action_n, layers):
        super().__init__()
        self.conv1 = nn.Conv2d(state_shape[0], layers[0], kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(layers[1], layers[2], kernel_size=2, stride=1)
        self.shape_after_convs = int(prod(cs(cs(cs(state_shape, self.conv1), self.conv2), self.conv3)))
        self.fc = nn.Linear(self.shape_after_convs, layers[3])
        self.exploit_head = FFNet(layers[3], layers[4:], action_n)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc.reset_parameters()
        self.exploit_head.reset_parameters()

    def _forward_impl(self, x0):
        x = F.leaky_relu(self.conv1(x0))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view((-1, self.shape_after_convs))
        x = F.leaky_relu(self.fc(x))
        out = self.exploit_head(x)
        return out

    def forward(self, x, a_dist=None, detach=True):
        if not detach: x.requires_grad = True
        q = self._forward_impl(x)
        if detach: q = q.detach()
        if a_dist is not None: return (a_dist * q).sum(-1)
        else:                  return q

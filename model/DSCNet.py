"""
By Xifeng Guo (guoxifeng1990@163.com), May 13, 2020.
All rights reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride` or `w_pad = w_in * stride - 1`, i.e., the "SAME" padding mode
    in Tensorflow. To determine the value of `w_pad`, we should pass it to this function.
    So the number of columns to delete:
        pad = 2*padding - output_padding = w_nopad - w_pad
    If pad is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    """

    def __init__(self, output_size):
        super(ConvTranspose2dSamePad, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = in_height - self.output_size[0]
        pad_width = in_width - self.output_size[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module(
                'conv%d' % i,
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2)
            )
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        # self.decoder = nn.Sequential()
        # channels = list(reversed(channels))
        # kernels = list(reversed(kernels))
        # sizes = [[12, 11], [24, 21], [48, 42]]
        # for i in range(len(channels) - 1):
        #     # Each layer will double the size of feature map
        #     self.decoder.add_module(
        #         'deconv%d' % (i + 1),
        #         nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2)
        #     )
        #     self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(sizes[i]))
        #     self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        # y = self.decoder(h)
        return h


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-4 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        # self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        # shape = z.shape
        z = z.view(x.shape[0], -1)  # shape=[n, d]
        # z_recon = self.self_expression(z)  # shape=[n, d]
        # z_recon_reshape = z_recon.view(shape)

        # x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return z

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = 0.5 * F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)  # just control the range, does not affect the optimization.
        return loss

class PRO_DSC(nn.Module):
    def __init__(self,hidden_dim, z_dim,channels=[1, 10, 20, 30],kernels=[5, 3, 3]):
        super().__init__()
        self.pre_feature = ConvAE(channels, kernels)
        self.subspace = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
        self.cluster = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
    def forward(self, x):
        
        pre_feature = self.pre_feature(x)
        pre_feature = pre_feature.view(pre_feature.shape[0],-1)
        Z = self.subspace(pre_feature)
        logits = self.cluster(pre_feature).float()
        Z = F.normalize(Z, 2)
        logits = F.normalize(logits, 2)

        return Z, logits
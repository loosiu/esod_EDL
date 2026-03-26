# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.common import YOLOv6Head, Conv
from utils.torch_utils import time_synchronized


class SparseTensor(object):
    def __init__(self, x, indices) -> None:
        self.features = x
        self.indices = indices


class SPConv2D1x1(nn.Module):
    # Sparse 2d-convolution layer 1x1
    def __init__(self, m: nn.Conv2d):
        super(SPConv2D1x1, self).__init__()
        assert m.kernel_size == (1, 1)

        self.weight = m.weight.flatten(1)
        self.bias = m.bias

    def forward(self, x, indices=None, to_dense=False):
        if indices is None:
            return F.linear(x, self.weight, self.bias)
        else:
            z = x.permute(0, 2, 3, 1).contiguous() # channel_last
            # bi, yi, xi = indices.T
            bi = indices[:, 0]
            yi = indices[:, 1]
            xi = indices[:, 2]
            y = F.linear(z[bi, yi, xi], self.weight, self.bias)
            if to_dense:
                # assert z.shape[-1] == y.shape[-1]
                z[bi, yi, xi] = y
                y = z.permute(0, 3, 1, 2).contiguous() # channel_first
            return y


class SPConv2Dkxk(nn.Module):
    # Sparse 2d-convolution layer kxk
    def __init__(self, m: nn.Conv2d):
        super(SPConv2Dkxk, self).__init__()
        self.weight = m.weight
        self.weight_flatten = m.weight.flatten(1)
        self.bias = m.bias
        self.kernel_size = m.kernel_size
        self.padding = m.padding
        self.stride = m.stride
        assert all([k == p * 2 + 1 for (k, p) in zip(m.kernel_size, m.padding)]), \
            f'Unsupported kernel size {m.kernel_size} and padding {m.padding}'
        assert m.stride == (1, 1), f'Unsupported stride {m.stride}'

    def forward(self, x, indices=None, to_dense=False):
        if indices is None:
            assert self.kernel_size == (1, 1) and self.padding == (0, 0)
            return F.linear(x, self.weight_flatten, self.bias)
        
        bs, c1, ny, nx = x.shape
        unfold = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        unfold = unfold.transpose(1, 2).view(bs, ny, nx, c1*np.prod(self.kernel_size))

        # bi, yi, xi = indices.T
        bi = indices[:, 0]
        yi = indices[:, 1]
        xi = indices[:, 2]
        z = F.linear(unfold[bi, yi, xi], self.weight_flatten, self.bias)

        if to_dense:
            x[bi, yi, xi] = z
            z = x
        
        return z


class SPConv(nn.Module):
    # Sparse convolution
    def __init__(self, conv: Conv):
        super(SPConv, self).__init__()
        self.conv = SPConv2Dkxk(conv.conv)
        if hasattr(conv, 'bn'):
            raise NotImplementedError
            self.bn = make_spbn(conv.bn)
        self.act = conv.act

    def forward(self, x, indices=None, to_dense=False):
        x = self.conv(x, indices, to_dense)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.act(x)
        return x

    def forward_dense(self, x):
        m = self.conv
        stem = F.conv2d(x, m.weight, m.bias, m.stride, m.padding)
        return self.act(stem)


class SPYOLOv6Head(nn.Module):
    def __init__(self, dense_head: YOLOv6Head):
        super(SPYOLOv6Head, self).__init__()
        self.na = dense_head.na
        self.nc = dense_head.nc
        self.sp_stem = SPConv(dense_head.stem)
        self.sp_cls_conv = SPConv(dense_head.cls_conv)
        self.sp_reg_conv = SPConv(dense_head.reg_conv)
        self.sp_cls_pred = SPConv2Dkxk(dense_head.cls_pred)
        self.sp_reg_pred = SPConv2Dkxk(dense_head.reg_pred)
        self.sp_obj_pred = SPConv2Dkxk(dense_head.obj_pred)

    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        assert not self.training
        
        # stem = self.sp_stem(x, indices, to_dense=True)
        stem = self.sp_stem.forward_dense(x)
        cls_feat = self.sp_cls_conv(stem, indices)
        reg_feat = self.sp_reg_conv(stem, indices)
        cls = self.sp_cls_pred(cls_feat).view(-1, self.na, self.nc)
        reg = self.sp_reg_pred(reg_feat).view(-1, self.na, 4)
        obj = self.sp_obj_pred(reg_feat).view(-1, self.na, 1)
        y_sp = torch.cat((reg, obj, cls), -1).view(-1, self.na*(4+1+self.nc))
        return SparseTensor(y_sp, indices)


class SPYOLOv5Head(nn.Module):
    def __init__(self, dense_head: nn.Conv2d):
        super(SPYOLOv5Head, self).__init__()
        self.sp_head = SPConv2D1x1(dense_head)
    
    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        assert not self.training
        y_sp = self.sp_head(x, indices)
        
        return SparseTensor(y_sp, indices)


if __name__ == '__main__':
    bs, c1, h, w = 2, 3, 6, 5
    c2 = 4
    k, s, p = 3, 1, 1

    with torch.no_grad():
        inp = torch.rand(bs, c1, h, w).cuda()
        conv = torch.nn.Conv2d(c1, c2, k, s, p, bias=True).cuda()

        # dummy
        unfold = F.unfold(inp*0.1, k, padding=p, stride=s)
        unfold = unfold.transpose(1, 2).view(bs, h, w, c1*k*k)
        # y = unfold @ conv.weight.flatten(1).T + conv.bias
        y = torch.matmul(unfold, conv.weight.flatten(1).transpose(0, 1)) + conv.bias


        t0 = time_synchronized()
        unfold = F.unfold(inp, k, padding=p, stride=s)
        unfold = unfold.transpose(1, 2).view(bs, h, w, c1*k*k)
        # y = unfold @ conv.weight.flatten(1).T + conv.bias
        y = torch.matmul(unfold, conv.weight.flatten(1).transpose(0, 1)) + conv.bias

        t1 = time_synchronized()
        y = y.permute(0, 3, 1, 2).contiguous()

        # dummy
        z = F.conv2d(inp*0.1, conv.weight, conv.bias, stride=s, padding=p)

        t2 = time_synchronized()
        z = F.conv2d(inp, conv.weight, conv.bias, stride=s, padding=p)
        t3 = time_synchronized()
        print(y.shape, z.shape)
        print(f'Error: {(y - z).abs().sum().item()}. Cost: {(t1 - t0)*1000:.1f}ms v.s. {(t3 - t2)*1000:.1f}ms')

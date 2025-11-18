# darts.py
# SC-NAS: Stabilizing DARTS via Dual Spectral Normalization
# Author: Pollob Hussain (Independent Researcher)
# Paper: https://arxiv.org/abs/xxxx.xxxxx (to be filled after submission)
# GitHub: https://github.com/GPollob/SC-NAS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# List of candidate operations (standard DARTS search space)
OPS = {
    'none'           : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3'   : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3'   : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect'   : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3'   : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5'   : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3'   : lambda C, stride, affine: DilConv(C, C, 3, stride, 1, 2, affine=affine),
    'dil_conv_5x5'   : lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 4, affine=affine),
}

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride, affine=True)
            # Operation-level Spectral Normalization (Op-SN)
            if 'pool' not in primitive and primitive != 'none':
                # Wrap conv layers with spectral_norm
                for layer in op.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        spectral_norm(layer, name='weight')
            self._ops.append(op)

    def forward(self, x, weights):
        # Architecture-parameter Spectral Normalization (Arch-SN)
        # L2-normalize weights before softmax → prevents explosion
        weights = weights / (weights.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        weights = F.softmax(weights, dim=-1)

        return sum(w * op(x) for w, op in zip(weights, self._ops))


# Standard building blocks (unchanged from original DARTS)
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            ReLUConvBN(C_in, C_in, kernel_size, stride, padding, affine),
            ReLUConvBN(C_in, C_out, kernel_size, 1, padding, affine),
        )
    def forward(self, x): return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(C_in, C_in, kernel_size, stride, padding=padding,
                                   dilation=dilation, groups=C_in, bias=False)),
            spectral_norm(nn.Conv2d(C_in, C_out, 1, padding=0, bias=False)),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x): return self.op(x)

class Identity(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1: return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = spectral_norm(nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

import torch
import torch.nn as nn
from .mlp import FiLM

class ConvNet(nn.Module):
    def __init__(self, in_channels, image_shape, filter_sizes, cnn_strides, cnn_depths, padding="valid", residual=True, dropout=0.0, norm=True, **kwargs):
        super().__init__()
        cnn_depths = [in_channels] + cnn_depths
        self.residual = residual
        self.norm = norm
        self._conv_layers = []
        self._nonlinear_layers = []
        self._norm_layers = []
        self.in_depths = cnn_depths[:-1]
        self.out_depths = cnn_depths[1:]
        for size, stride, in_depth, out_depth in zip(filter_sizes, cnn_strides, self.in_depths, self.out_depths):
            self._norm_layers.append(nn.GroupNorm(num_groups=1, num_channels=in_depth))
            self._conv_layers.append(nn.Conv2d(in_depth, out_depth, size, stride=stride, padding=padding))
            self._nonlinear_layers.append(nn.ReLU())
        self.out_shape = get_feature_shape((in_channels, *image_shape), self._conv_layers)
        self._norm_layers = nn.ModuleList(self._norm_layers)
        self._conv_layers = nn.ModuleList(self._conv_layers)
        self._nonlinear_layers = nn.ModuleList(self._nonlinear_layers)
        self._dropout = nn.Dropout(p = dropout)

    def __call__(self, x):
        # x is (B, C, H, W)
        B, C, H, W = x.shape
        for conv_layer, nonlinear, norm, in_depth, out_depth in zip(self._conv_layers, self._nonlinear_layers, self._norm_layers, self.in_depths, self.out_depths):
            x_original = x
            if self.norm:
                x = norm(x)
            x = conv_layer(x)
            x = nonlinear(x)
            if self.residual and in_depth == out_depth:
                x = x_original + x
        return self._dropout(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, image_shape, cnn_depth, filter_size=3, cnn_stride=1, num_layers=2, padding="valid", dropout=0.0, norm=True, cond_dim=0, **kwargs):
        super().__init__()
        self.norm = norm
        self.num_layers = num_layers
        self._conv_layers = []
        self._nonlinear_layers = []
        self._norm_layers = []

        # input layer
        if in_channels != cnn_depth:
            self._input_layer = nn.Conv2d(in_channels, cnn_depth, 1, stride=1, padding="same")
        else:
            self._input_layer = None

        for _ in range(num_layers):
            self._norm_layers.append(nn.GroupNorm(num_groups=1, num_channels=cnn_depth))
            self._conv_layers.append(nn.Conv2d(cnn_depth, cnn_depth, filter_size, stride=cnn_stride, padding=padding))
            self._nonlinear_layers.append(nn.ReLU())
        
        self.out_shape = get_feature_shape((in_channels, *image_shape), self._conv_layers if self._input_layer is None else [self._input_layer] + self._conv_layers)
        self._cond_layer = FiLM(cond_dim, self.out_shape[0]) if cond_dim>0 else None

        self._norm_layers = nn.ModuleList(self._norm_layers)
        self._conv_layers = nn.ModuleList(self._conv_layers)
        self._nonlinear_layers = nn.ModuleList(self._nonlinear_layers)
        self._dropout = nn.Dropout(p = dropout)

    def __call__(self, x):
        if not (self._cond_layer is None):
            x, cond_vec = x
        # x is (B, C, H, W)
        if not self._input_layer is None:
            x = self._input_layer(x)
        x_skip = x
        for i, (conv_layer, nonlinear, norm) in enumerate(zip(self._conv_layers, self._nonlinear_layers, self._norm_layers)):
            if self.norm:
                x = norm(x)
            x = conv_layer(x)
            if (not self._cond_layer is None) and (i == (self.num_layers-1)):
                x = self._cond_layer(x, cond_vec)
            x = nonlinear(x)
        self._dropout(x)
        return (x + x_skip) if self._cond_layer is None else (x + x_skip, cond_vec)

def get_feature_size(input_shape, conv_layers):
    # returns flattened size
    x = torch.zeros(input_shape)
    for conv_layer in conv_layers:
        x = conv_layer(x)
    return torch.numel(x)

def get_feature_shape(input_shape, conv_layers):
    # returns flattened size
    x = torch.zeros(input_shape)
    for conv_layer in conv_layers:
        x = conv_layer(x)
    return x.shape
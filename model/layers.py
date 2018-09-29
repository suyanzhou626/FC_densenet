import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense_layer(nn.Module):
    def __init__(self, input_depth, n_filters=16, filter_size=3, dropout_p=0.2):
        super(Dense_layer, self).__init__()

        self.BN = nn.BatchNorm2d(input_depth)
        self.conv = nn.Conv2d(input_depth, n_filters,
                              filter_size, 1, padding=(filter_size-1)//2)
        self.dropout = nn.Dropout(dropout_p)

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = F.relu(self.BN(input))
        out = self.conv(out)
        out = self.dropout(out)

        return out

class Transition_Down(nn.Module):
    def __init__(self, input_depth, output_depth, dropout_p=0.2):
        super(Transition_Down, self).__init__()

        self.BN = nn.BatchNorm2d(input_depth)
        self.conv = nn.Conv2d(input_depth, output_depth, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.pool = nn.MaxPool2d(2, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = F.relu(self.BN(input))
        out = self.conv(out)
        out = self.dropout(out)
        out = self.pool(out)

        return out


class Transition_Up(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(Transition_Up, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_depth, output_depth, 3, 2, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.deconv(input)
        return out

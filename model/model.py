import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.layers import Dense_layer, Transition_Down, Transition_Up


class DenseBlock(nn.Module):
    def __init__(self, input_depth, n_filters=16, filter_size=3, n_layers_per_block=5, dropout_p=0.2):
        super(DenseBlock, self).__init__()
        self.dense_list = nn.ModuleList([])

        self.num_layers = n_layers_per_block

        for i in range(n_layers_per_block):
            dense_layer = Dense_layer(
                input_depth, n_filters, filter_size, dropout_p)
            self.dense_list.append(dense_layer)

            input_depth += n_filters

    def forward(self, input):
        concat_list = []

        input = input
        for i in range(self.num_layers):
            output = self.dense_list[i](input)
            concat_list.append(output)

            input = torch.cat((output, input), dim=1)

        output = torch.cat(concat_list, dim=1)
        return output


class Network(nn.Module):
    def __init__(self, n_layers_list, n_pool, input_depth=3, n_first_conv_filters=48, n_filters=16, filter_size=3, dropout_p=0.2, n_classes=12):
        super(Network, self).__init__()

        self.n_pool = n_pool
        self.n_layers_list = n_layers_list
        self.depth_list = []
        self.dense_blocks = nn.ModuleList([])
        self.TD = nn.ModuleList([])
        self.TU = nn.ModuleList([])

        assert(len(n_layers_list) == 2 * n_pool + 1)

        self.first_conv = nn.Conv2d(
            input_depth, n_first_conv_filters, filter_size, 1, (filter_size-1)//2)

        ###########################
        # First Convolution layer #
        ###########################

        input_depth = n_first_conv_filters  # 48

        count = 0

        #############################
        ##### DownSampling path #####
        #############################
        for i in range(n_pool):
            # Make i-th Dense block
            dense_block = DenseBlock(
                input_depth, n_filters, filter_size, n_layers_list[i], dropout_p)

            # After dense block, we concate nate with output and caching it
            input_depth = input_depth + n_layers_list[i] * n_filters
            # 112, 192, 304, 464, 656

            # Then Transtion Down block
            transition_down = Transition_Down(
                input_depth, input_depth, dropout_p)

            self.dense_blocks.append(dense_block)
            self.depth_list.append(input_depth)
            self.TD.append(transition_down)

        #################################
        #### Bottle Neck Dense block ####
        #################################
        self.dense_blocks.append(DenseBlock(
            input_depth, n_filters, filter_size, n_layers_list[self.n_pool], dropout_p))
        input_depth = input_depth + n_filters * \
            n_layers_list[self.n_pool]  # 896
        self.depth_list.append(input_depth)

        # print(self.depth_list)   # For debugging
        # May be self.depth_list is [112, 192, 304, 464, 656, 896]

        self.depth_list = self.depth_list[::-1]
        # May be self.depth_list is [896, 656, 464, 304, 192, 112]

        #############################
        ###### UpSampling path ######
        #############################
        for idx, i in enumerate(range(self.n_pool+1, len(n_layers_list), 1)):
            n_filters_keep = n_filters * \
                n_layers_list[i-1]  # 240, 192, 160, 112, 80
            transtion_up = Transition_Up(n_filters_keep, n_filters_keep)

            dense_block = DenseBlock(
                self.depth_list[idx], n_filters, filter_size, n_layers_list[i], dropout_p)

            # 1088, 816, 578, 384, 256
            # input_depth = self.depth_list[idx_depth_list] + \
            #    n_layers_list[i] * n_filters

            # print(input_depth)  # For debugging

            self.TU.append(transtion_up)
            self.dense_blocks.append(dense_block)

        self.last_conv = nn.Conv2d(
            n_layers_list[-1]*n_filters, n_classes, 1, 1)

    def forward(self, input):

        cache_list = []

        # First conv
        first_conv = self.first_conv(input)
        _in = first_conv

        #############################
        ##### DownSampling path #####
        #############################
        for i in range(self.n_pool):
            out = self.dense_blocks[i](_in)
            out = torch.cat((out, _in), dim=1)
            cache_list.append(out)  # 112, 192, 304, 464, 656
            out = self.TD[i](out)

            _in = out

        # print(count)         #For debugging
        # count will be 5

        #################################
        #### Bottle Neck Dense block ####
        #################################
        out = self.dense_blocks[self.n_pool](_in)  # 240

        _in = out

        #count -= 1  # count will be 4
        cache_list = cache_list[::-1]  # 656, 464, 304, 192, 112

        #############################
        ###### UpSampling path ######
        #############################
        for idx, i in enumerate(range(self.n_pool+1, len(self.dense_blocks), 1)):
            out = self.TU[idx](_in)  # 240
            out = torch.cat((out, cache_list[idx]), dim=1)
            out = self.dense_blocks[i](out)
            _in = out

        out = self.last_conv(_in)
        out = F.softmax(out, dim=1)

        return out

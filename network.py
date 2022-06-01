import torch
import torch.nn as nn
import numpy as np

from util import util


def parse_cfg(filename):
    """
    parse configuration files into list of layer dictionary

    :param filename: path of configuration file
    :return: blocks containing layers
    """
    file = open(filename, 'r')
    # get each line of the file
    lines = file.read().split('\n')

    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.lstrip().rstrip() for x in lines]

    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[':  # deal with layer type information
            if len(block) > 0:
                # append the block and reinitialize it in case it already contains information
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].lstrip().rstrip()
        else:  # deal with necessary parameters for the layer
            param, val = line.split('=')
            param = param.rstrip()
            val = val.lstrip()
            block[param] = val
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    create module list from the configuration block

    :param blocks: list of layer dictionaries
    :return: tuple of network information and module list
    """
    module_list = nn.ModuleList()
    prev_filters = 3
    filter_list = []

    net_info = blocks[0]

    # iterate the layer blocks from position 1
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            try:
                batch = int(block['batch_normalize'])
                bias = 0
            except:
                batch = 0
                bias = 1

            filters = int(block['filters'])
            kernel = int(block['size'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            activation = block['activation']

            if pad:
                padding_size = (kernel - 1) // 2
            else:
                padding_size = 0

            conv = nn.Conv2d(prev_filters, filters, kernel, stride, padding_size, bias=bias)
            module.add_module('conv_{}'.format(index), conv)

            if batch:
                batch_layer = nn.BatchNorm2d(filters)
                module.add_module('batch_{}'.format(index), batch_layer)

            if activation == 'leaky':
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('activation_{}'.format(index), activation_layer)


        elif block['type'] == 'shortcut':
            shortcut_layer = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut_layer)

        elif block['type'] == 'upsample':
            scale = int(block['stride'])
            upsample_layer = nn.Upsample(scale_factor=scale, mode='nearest')
            module.add_module('upsample_{}'.format(index), upsample_layer)

        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(item) for item in layers]

            start = layers[0]

            try:
                end = layers[1]
            except:  # in case only one parameter is provided for the route layer
                end = 0

            if start > 0:
                # calculate the relative position if absolute index is provided
                start = start - index
            if end > 0:
                end = end - index

            if end < 0:
                filters = filter_list[index + start] + filter_list[index + end]
            else:
                filters = filter_list[index + start]

            route_layer = EmptyLayer()
            module.add_module('route_{}'.format(index), route_layer)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(item.lstrip().rstrip()) for item in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(item.lstrip().rstrip()) for item in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            yolo_layer = DetectionLayer(anchors)
            module.add_module('yolo_{}'.format(index), yolo_layer)

        # append the new created module
        module_list.append(module)
        # keep track of the filter size
        prev_filters = filters
        filter_list.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
    """
    underlying network of yolo v3
    """

    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        output_list = {}
        create = 0

        for index, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional' or block['type'] == 'upsample':
                x = self.module_list[index](x)

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(item) for item in layers]

                start = layers[0]

                try:
                    end = layers[1]
                except:
                    end = 0

                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index

                if end < 0:
                    # concatenate two layers if two parameters are provided for route layer
                    x = torch.cat((output_list[index + start], output_list[index + end]), 1)
                else:  # return the given layer if only one paramtere provided
                    x = output_list[index + start]

            elif block['type'] == 'shortcut':
                shortcut_from = int(block['from'])
                if shortcut_from > 0:
                    shortcut_from = shortcut_from - index

                # add the given layer with current layer for shortcut layer
                x = output_list[index - 1] + output_list[index + shortcut_from]

            elif block['type'] == 'yolo':
                anchors = self.module_list[index][0].anchors
                input_dim = int(self.net_info['height'])
                class_num = int(block['classes'])

                x = x.data
                # transform the output of the network
                x = util.transform_prediction(x, input_dim, anchors, class_num)

                if not create:
                    detection = x
                    create = 1
                else:
                    detection = torch.cat((detection, x), 1)

            output_list[index] = x

        return detection

    def load_weights(self, weightfile):
        """
        load network using provided weights

        :param weightfile: path to the weight file
        """
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally, copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

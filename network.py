import torch
import torch.nn as nn

import util
from util import *

import cv2
from torch.autograd import Variable


def parse_cfg(filename):
    file = open(filename, 'r')
    lines = file.read().split('\n')

    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.lstrip().rstrip() for x in lines]

    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[':
            if len(block) > 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].lstrip().rstrip()
        else:
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
    module_list = nn.ModuleList()
    prev_filters = 3
    filter_list = []

    net_info = blocks[0]

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            try:
                batch = int(block['batch_normalize'])
            except:
                batch = 0
            filters = int(block['filters'])
            kernel = int(block['size'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            activation = block['activation']
            bias = -1

            if pad:
                padding_size = (kernel - 1) // 2
                bias = 0
            else:
                padding_size = 0
                bias = 1

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
            upsample_layer = nn.Upsample(scale_factor=scale)
            module.add_module('upsample_{}'.format(index), upsample_layer)

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
            anchors = [(anchors[i], anchors[i + 1]) for i in range(int(len(anchors) / 2))]
            anchors = [anchors[i] for i in mask]

            yolo_layer = DetectionLayer(anchors)
            module.add_module('yolo_{}'.format(index), yolo_layer)

        module_list.append(module)
        prev_filters = filters
        filter_list.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
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
                    x = torch.cat((output_list[index + start], output_list[index + end]), 1)
                else:
                    x = output_list[index + start]

            elif block['type'] == 'shortcut':
                shortcut_from = int(block['from'])
                if shortcut_from > 0:
                    shortcut_from = shortcut_from - index

                x = output_list[index - 1] + output_list[index + shortcut_from]

            elif block['type'] == 'yolo':
                anchors = self.module_list[index][0].anchors
                input_dim = int(self.net_info['height'])
                class_num = int(block['classes'])

                x = x.data
                x = util.transform_detection(x, input_dim, anchors, class_num)

                if not create:
                    detection = x
                    create = 1
                else:
                    detection = torch.cat((detection, x), 1)

            output_list[index] = x

        return detection


# blocks = parse_cfg('./cfg/yolov3.cfg')
# net_info, modules = create_modules(blocks)
# print(modules[106][0].anchors)
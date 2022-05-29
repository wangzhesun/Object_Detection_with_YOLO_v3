import torch
import numpy as np


def transform_detection(prediction, input_dim, anchors, class_num):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + class_num
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid,grid)
    x_offset = torch.FloatTensor(x).view(-1, 1)
    y_offset = torch.FloatTensor(y).view(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] = torch.sigmoid(prediction[:, :, :2])
    prediction[:, :, :2] = prediction[:, :, :2] + x_y_offset

    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = np.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 4:] = torch.sigmoid(prediction[:, :, 4:])

    prediction[:, :, :4] *= stride

    return prediction

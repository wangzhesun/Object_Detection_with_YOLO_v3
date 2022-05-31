import torch
import numpy as np
import cv2 as cv


def transform_prediction(prediction, input_dim, anchors, class_num):
    """
    transform the raw output of the net to a 2-D form for easier processing

    :param prediction: raw output
    :param input_dim: size of input image
    :param anchors: anchors used
    :param class_num: number of classes
    :return: new output (2-D tensor)
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + class_num
    num_anchors = len(anchors)

    # rearrange the prediction to 2-D tensor, each row representing a bounding box for an anchor
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x).view(-1, 1)
    y_offset = torch.FloatTensor(y).view(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] = torch.sigmoid(prediction[:, :, :2])
    prediction[:, :, :2] = prediction[:, :, :2] + x_y_offset

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 4:] = torch.sigmoid(prediction[:, :, 4:])

    # recover the feature map coordinate to real image coordinate
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    """
    get unique elements from given input

    :param tensor: tensor provided
    :return: unique element in the given tensor
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    returns the IoU of two bounding boxes

    :param box1: a single bounding box
    :param box2: a list of bounding box
    :return: the IoU between the first bounding box and every bounding box in second parameter
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def NMS(prediction, nms_threshold):
    """
    perform non-maximum suppression on the prediction using provided threshold

    :param prediction: network prediction
    :param nms_threshold: threshold for the NMS
    :return: non-maximum suppressed prediction
    """
    class_size = prediction.shape[0]
    for i in range(class_size):
        try:
            ious = bbox_iou(prediction[i].unsqueeze(0), prediction[i + 1:])
        except ValueError:
            break

        except IndexError:
            break

        # Zero out all the detections that have IoU > treshhold
        iou_mask = (ious < nms_threshold).float().unsqueeze(1)
        prediction[i + 1:] *= iou_mask

        # Remove the non-zero entries
        non_zero_ind = torch.nonzero(prediction[:, 4]).squeeze()
        prediction = prediction[non_zero_ind].view(-1, 7)

    return prediction


def write_results(prediction, conf_thresh, nms_thresh, num_class):
    """
    perform objectness score thresholding and non-maximum suppression to the prediction

    :param prediction: input prediction
    :param conf_thresh: threshold for objectness score
    :param nms_thresh: threshold for non-maximum suppression
    :param num_class: number of classes
    :return: final predictions
    """
    batch_size = prediction.size(0)
    write = False

    conf_mask = (prediction[:, :, 4] > conf_thresh).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # iterate through all batch images
    for index in range(batch_size):
        pred_batch = prediction[index]

        # replace the score for each class with the maximum score and the corresponding index
        max_score, max_index = torch.max(pred_batch[:, 5:5 + num_class], 1)
        max_score = max_score.float().unsqueeze(1)
        max_index = max_index.float().unsqueeze(1)
        pred_batch = torch.cat((pred_batch[:, :5], max_score, max_index), 1)

        # get rid of zero confidence entry
        nonzero_index = torch.nonzero(pred_batch[:, 4])
        try:
            pred_batch_ = pred_batch[nonzero_index.squeeze(), :].view(-1, 7)
        except:
            continue

        if pred_batch_.shape[0] == 0:
            continue

        # get unique classes
        unique_classes = unique(pred_batch_[:, -1])

        # iterate through all unique classes
        for uclass in unique_classes:
            # get predictions with specific classes
            class_mask = (pred_batch_[:, -1] == uclass).float().unsqueeze(1)
            class_mask = pred_batch_ * class_mask
            class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()
            pred_batch_class = pred_batch_[class_mask_index].view(-1, 7)

            # sort the prediction using the class score
            class_sort_index = torch.sort(pred_batch_class[:, 4], descending=True)[1]
            pred_batch_class = pred_batch_class[class_sort_index]

            # perform nms to eliminate unnecessary ones
            pred_batch_class = NMS(pred_batch_class, nms_thresh)

            batch_ind = pred_batch_class.new(pred_batch_class.size(0), 1).fill_(index)
            out = torch.cat((batch_ind, pred_batch_class), 1)

            if not write:
                output = out
                write = True
            else:
                output = torch.cat((output, out), 0)

    try:
        return output
    except:
        return 0


def load_classes(namesfile):
    """
    get the names of classes

    :param namesfile: path to the class file
    :return: class names
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    """
    resize the input image with unchanged aspect ratio using padding

    :param img: input image
    :param inp_dim: input dimension of the network
    :return: the aligned input images
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w,
    :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    prepare image for inputting to the neural network using the letterbox_image function

    :param img: the original image
    :param inp_dim: input dimension of the network
    :return: the network input image
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

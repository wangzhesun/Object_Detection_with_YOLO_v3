from __future__ import division

import time
import torch
from torch.autograd import Variable
from util import util
import os
from network import Darknet
import pickle as pkl
import pandas as pd
import cv2 as cv
import random


def write_img(x, results):
    """
    draw bounding boxes and prediction class on images

    :param x: prediction output
    :param results: original images
    :return: images with prediction boxes
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])

    cv.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 1)

    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    cv.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)

    cv.putText(img, label, (int(c1[0]), int(c1[1] + t_size[1] + 4)), cv.FONT_HERSHEY_PLAIN, 1,
               [225, 255, 255], 1)
    return img


def YoloDetect(src, det, batch_size=1, confidence=0.5, nms_thresh=0.4):
    """
    run yolo v3 on the input images provided by src and output results to det

    :param src: path to the source images: will deal with all images if the path is a directory
    :param det: destination path for output images
    :param batch_size: batch size, default = 1
    :param confidence: confidence threshold, default = 0.5
    :param nms_thresh: non-maximum suppression threshold, default = 0.4
    """
    read_dir = time.time()
    try: # try if the src is a directory path
        imlist = [src+'/'+img for img in os.listdir(src)]
    except NotADirectoryError: # try file path if it is not a directory
        imlist = [src]
    except FileNotFoundError: # invalid path if it is neither directory nor file
        print("No file or directory with the name {}".format(src))
        exit()

    # make the destination directory if not exist already
    if not os.path.exists(det):
        os.makedirs(det)

    load_batch = time.time()
    loaded_ims = [cv.imread(x) for x in imlist]

    # preprocessing input images
    im_batches = list(map(util.prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in
                      range(num_batches)]

    write = 0

    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        with torch.no_grad():
            prediction = model(Variable(batch))

        prediction = util.write_results(prediction, confidence, nms_thresh, num_classes)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(
                    imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                     (end - start) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        # transform the attribute from index in batch to index in imlist
        prediction[:, 0] += i * batch_size

        if not write:  # If we haven't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(
                imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                 (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()
    class_load = time.time()
    draw = time.time()

    # draw bounding boxes
    list(map(lambda x: write_img(x, loaded_ims), output))

    imlist = [img.split('/')[-1] for img in imlist]

    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(det, x.split("\\")[-1]))

    list(map(cv.imwrite, det_names, loaded_ims))

    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)",
                                   output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")


if __name__ == '__main__':
    num_classes = 80
    classes = util.load_classes("data/coco.names.txt")

    # Set up the neural network
    print("Loading network.....")
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("weights/yolov3.weights")
    print("Network successfully loaded")

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    colors = pkl.load(open("pallete", "rb"))

    # run yolo v3 detection
    # change the first parameter to the source path, the second to the destination path
    YoloDetect('./imgs', './det')
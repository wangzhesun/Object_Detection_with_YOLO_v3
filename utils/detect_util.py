from __future__ import division

import time
import torch
from torch.autograd import Variable
from utils import model_util as util
import os
from network import Darknet
import pickle as pkl
import pandas as pd
import cv2 as cv
import random


def write_img(x, results, colors, classes):
    """
    draw bounding boxes and prediction class on images

    :param x: prediction output
    :param results: original images
    :param colors: color configuration file
    :param classes: class configuration file
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


def write_vid(x, img, colors, classes):
    """
    draw bounding boxes and prediction class on video frames

    :param x: prediction output
    :param img: original video
    :param colors: color configuration file
    :param classes: class configuration file
    :return: video with prediction boxes
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)
    cv.putText(img, label, (int(c1[0]), int(c1[1] + t_size[1] + 4)), cv.FONT_HERSHEY_PLAIN, 1,
               [225, 255, 255], 1)
    return img


def detect_img(model, src, det, batch_size, confidence, nms_thresh, colors, classes, num_classes,
               inp_dim):
    """
    helper function to detect object in images with yolo

    :param model: detection model
    :param src: path to the source images: process all images if the path is a directory
    :param det: destination path for output images/videos
    :param batch_size: batch size, default = 1
    :param confidence: confidence threshold, default = 0.5
    :param nms_thresh: non-maximum suppression threshold, default = 0.4
    :param colors: color configuration file
    :param classes: class configuration file
    :param num_classes: number of classes that can be detected
    :param inp_dim: dimension of the input
    """
    read_dir = time.time()
    try:  # try if the src is a directory path
        imlist = [src + '/' + img for img in os.listdir(src)]
    except NotADirectoryError:  # try file path if it is not a directory
        imlist = [src]
    except FileNotFoundError:  # invalid path if it is neither directory nor file
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
    list(map(lambda x: write_img(x, loaded_ims, colors, classes), output))

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


def detect_vid(model, src, det, confidence, nms_thresh, colors, classes, num_classes, inp_dim):
    """
    helper function to detect object in videos with yolo

    :param model: object detection model
    :param src: path to the source images: process all images if the path is a directory
    :param det: destination path for output images/videos
    :param confidence: confidence threshold, default = 0.5
    :param nms_thresh: non-maximum suppression threshold, default = 0.4
    :param colors: color configuration file
    :param classes: class configuration file
    :param num_classes: number of classes that can be detected
    :param inp_dim: dimension of the input
    """
    cap = cv.VideoCapture(src)

    assert cap.isOpened(), 'Cannot capture source'

    # make the destination directory if not exist already
    if not os.path.exists(det):
        os.makedirs(det)

    # set up output object
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    output_vid_name = '{}/det_{}.avi'.format(det, src.split('/')[-1].split('.')[0])

    out = cv.VideoWriter(output_vid_name, fourcc, 20.0, (cols, rows))

    # Get frame count
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # prepare input video
            img, orig_im, dim = util.prep_video(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            with torch.no_grad():
                output = model(Variable(img))

            output = util.write_results(output, confidence, nms_thresh, num_classes)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(
                    frames / (time.time() - start)) + "  frame: " + str(frames) + "/" + str(
                    n_frames))

                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            list(map(lambda x: write_vid(x, orig_im, colors, classes), output))

            # write output
            out.write(orig_im)
            frames += 1
            print("FPS of the video is {:5.2f}".format(
                frames / (time.time() - start)) + "  frame: " + str(frames) + "/" + str(
                n_frames))
        else:
            break

    out.release()


def yolo_detect(model, src, det, batch_size=1, confidence=0.5, nms_thresh=0.4, img=1, colors=None,
                classes=None, num_classes=None, inp_dim=None):
    """
    run yolo v3 on the input images/videos provided by src and output results to demo

    :param model: object detection model
    :param src: path to the source images/videos. In the case dealing with images, process all
                images if the path is a directory
    :param det: destination path for output images/videos
    :param batch_size: batch size, default = 1
    :param confidence: confidence threshold, default = 0.5
    :param nms_thresh: non-maximum suppression threshold, default = 0.4
    :param img: flag img = 1 indicates dealing with images; = 0 indicates dealing with videos
    :param colors: color configuration file
    :param classes: class configuration file
    :param num_classes: number of classes that can be detected
    :param inp_dim: dimension of the input
    """
    if img == 1:  # flag img == 1 indicates we're dealing with images
        detect_img(model, src, det, batch_size, confidence, nms_thresh, colors, classes,
                   num_classes, inp_dim)
    else:  # flag img == 0 indicates we're dealing with images
        detect_vid(model, src, det, confidence, nms_thresh, colors, classes, num_classes, inp_dim)


def detect(input, output, img=1, num_classes=80, classes_path="configurations/coco.names.txt",
           network_config="configurations/yolov3.cfg", weight_path="weights/yolov3.weights"):
    """
    perform object detection

    :param input: the path of input image/video
    :param output: the path directory of the output
    :param img: flag indicating whether the input is an image or a video
    :param num_classes: the number of objects that can be detected
    :param classes_path: the path of the class configuration file
    :param network_config: the path of the network configuration file
    :param weight_path: the path of the pre-trained weight file
    """
    classes = util.load_classes(classes_path)

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(network_config)
    model.load_weights(weight_path)
    print("Network successfully loaded")

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    colors = pkl.load(open("configurations/pallete", "rb"))

    yolo_detect(model, input, output, img=img, colors=colors, classes=classes,
                num_classes=num_classes, inp_dim=inp_dim)

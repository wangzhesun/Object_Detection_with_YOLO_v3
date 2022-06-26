# Object Detection with YOLO v3

This repository contains a PyTorch implementation of the YOLO v3 object detector. The detector accepts
either images or videos as input and outputs images or videos with bounding boxes of detected objects.
This project is highly motivated by the tutorial written by Ayoosh Kathuria and is a reimplementation 
of his posts, which can be found [here](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/).
The YOLO network of this project is initialized with official pre-trained weights trained using the COCO 
dataset (80 object categories).

## Demos
Demos of this project can be found in the directory `demos`. the `.jpg` files are output images of the 
object detector, and the `.avi` file is the output video file.

## Usage
To use the object detection system of this project, you need to first download the
pre-trained weight files from [here](https://pjreddie.com/media/files/yolov3.weights) and place it
in the directory named `weights` (with a place-holder file inside). Then, you are ready to perform object detection 
on your own image or video using the `detect.py` script.

Examples of the `detetc.py` usage are shown as follow:

-- if dealing with images --
```
detect(input='./imgs/beach.jpg', output='./demos', img=1)
```
-- if dealing with videos --
```
detect(input='./vids/test.mp4', output='./demos', img=0)
```

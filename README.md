# Object Detection with YOLO v3

This repository contains a PyTorch implementation of yolov3 object detector. The detector accepts
either images or videos as input and outputs images/videos with bounding boxes of detected objects.
This project is highly motivated by the tutorial written by Ayoosh Kathuria and is a reimplementation 
of his posts, which can be found [here](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/).
The yolo network of this project is initialized with official pre-trained weights trained using COCO 
dataset (80 object categories). The weight file can be downloaded from [here](https://pjreddie.com/media/files/yolov3.weights).
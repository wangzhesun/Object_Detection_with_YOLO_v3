from utils.detect_util import detect

if __name__ == '__main__':
    # run yolo v3 detection
    # change the first parameter to the source path, the second to the path of destination directory
    # change flag img to 1 if dealing with images, change it to 0 if dealing with video

    detect(input='imgs/beach.jpg', output='./det', img=1)
    # detect(input='vids/test2.mp4', output='./det', img=0)

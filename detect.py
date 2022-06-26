from utils.detect_util import detect

if __name__ == '__main__':
    # run yolo v3 detection
    # change the first parameter to the source path, the second to the path of destination directory
    # change flag img to 1 if dealing with images, change it to 0 if dealing with video

    detect(input='input_path', output='directory_path_to_store_output', img=1)

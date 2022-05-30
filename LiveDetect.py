#!/usr/bin/env python

"""
Example code for live video processing
Also multithreaded video processing sample using opencv 3.4

Usage:
   python testcv_mt.py {<video device number>|<video file name>}

   Use this code as a template for live video processing

   Also shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts: (video display window must be selected

   ESC - exit
   space - switch between multi and single threaded processing
   d - running difference of current and previous image
   e - displays canny edges
   f - displays raw frames
   v - write video output frames to file "vid_out.avi"
"""

from collections import deque
from multiprocessing.pool import ThreadPool

# import the necessary packages
import cv2 as cv
import numpy as np

from util import video_util
from util import util
from time import perf_counter, sleep


# used to execute process_frame when in non threaded mode
class DummyTask:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def ready():
        return True

    def get(self):
        return self.data


# initialize global variables
frame_counter = 0
show_frames = True
diff_frames = False
show_edges = False
vid_frames = False


# this routine is run each time a new video frame is captured
def process_frame(_frame, _prevFrame, _currCount):
    if not show_frames and show_edges:  # edges alone
        edges = cv.Canny(_frame, 100, 200)
        thisFrame = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
    elif show_frames and show_edges:  # edges and frames
        edges = cv.Canny(_frame, 100, 200)
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
        thisFrame = cv.add(_frame, edges)
    else:  # current frame
        thisFrame = _frame.copy()

    if diff_frames:
        # compute absolute difference between the current and previous frame
        difframe = cv.absdiff(thisFrame, _prevFrame)
        # save current frame as previous
        _prevFrame = thisFrame.copy()
        # set the current frame to the difference image
        thisFrame = difframe.copy()
    else:
        # save current frame as previous
        _prevFrame = thisFrame.copy()

    return thisFrame, _prevFrame, _currCount


# create a video capture object
# noinspection DuplicatedCode
def create_capture(source=0):
    # parse source name (defaults to 0 which is the first USB camera attached)

    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass

    params = dict(s.split('=') for s in chunks[1:])

    # video capture object defined on source

    timeout = 100
    _iter = 0
    _cap = cv.VideoCapture(source)
    while (_cap is None or not _cap.isOpened()) & (_iter < timeout):
        sleep(0.1)
        _iter = _iter + 1
        _cap = cv.VideoCapture(source)

    if _iter == timeout:
        print('camera timed out')
        return None
    else:
        print(_iter)

    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        _cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        _cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    if _cap is None or not _cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return None

    return _cap


# main program
if __name__ == '__main__':
    import sys

    # print in the program shell window the text at the beginning of the file
    print(__doc__)

    # if there is no argument in the program invocation default to camera 0
    # noinspection PyBroadException
    # try:
    #     fn = sys.argv[1]
    # except:
    #     fn = 0
    if len(sys.argv) < 2:
        fn = 0
    else:
        fn = sys.argv[1]

    # grab initial frame, create window
    cv.waitKey(1) & 0xFF
    cap = video_util.create_capture(fn)
    ret, frame = cap.read()
    frame_counter += 1
    height, width, channels = frame.shape
    prevFrame = frame.copy()
    cv.namedWindow("video")

    # Create video of Frame sequence -- define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    cols = np.int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = np.int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_out = cv.VideoWriter('vid_out.avi', fourcc, 20.0, (cols, rows))

    # Set up multiprocessing
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = True
    onOff = False

    # initialize time variables
    latency = util.StatValue()
    frame_interval = util.StatValue()
    last_frame_time = perf_counter()

    # main program loop
    while True:
        while len(pending) > 0 and pending[0].ready():  # there are frames in the queue
            res, prevFrame, t0 = pending.popleft().get()
            latency.update(perf_counter() - t0)
            # plot info on threading and timing on the current image
            # comment out the next 3 lines to skip the plotting
            util.draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            util.draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            util.draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            if vid_frames:
                vid_out.write(res)
            # show the current image
            cv.imshow('video', res)

        if len(pending) < threadn:  # fewer frames than thresds ==> get another frame
            # get frame
            ret, frame = cap.read()
            frame_counter += 1
            t = perf_counter()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), prevFrame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, prevFrame, t))
            pending.append(task)

        # check for a keypress
        key = cv.waitKey(1) & 0xFF

        # threaded or non threaded mode
        if key == ord(' '):
            threaded_mode = not threaded_mode
        # toggle edges
        if key == ord('e'):
            show_edges = not show_edges
            if not show_edges and not show_frames:
                show_frames = True
        # toggle frames
        if key == ord('f'):
            show_frames = not show_frames
            if not show_frames and not show_edges:
                show_frames = True
        # image difference mode
        if key == ord('d'):
            diff_frames = not diff_frames
        # ESC terminates the program
        if key == ord('v'):
            vid_frames = not vid_frames
            if vid_frames:
                print("Frames are being output to video")
            else:
                print("Frames are not being output to video")

        # ESC terminates the program
        if key == 27:
            # release video capture object
            cap.release()
            # release video output object
            vid_out.release()
            cv.destroyAllWindows()
            break

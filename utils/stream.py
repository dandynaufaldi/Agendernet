import datetime
import cv2
from threading import Thread
from queue import Queue


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (datetime.datetime.now() - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src: str="../data/[MV] Fortune Cookie in Love (Fortune Cookie Yang Mencinta) - JKT48.mp4"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.q = Queue(maxsize=0)
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self._numFrames = 0
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self._numFrames += 1
            # self.q.put(self.frame)

    def read(self):
        # return the frame most recently read
        # res = self.q.get()
        # self.q.task_done()
        # return res
        return self.grabbed, self.frame

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # while not self.q.empty():
        #     _ = self.q.get()
        #     self.q.task_done()
        # self.q.join()

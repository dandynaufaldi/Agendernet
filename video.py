import cv2
import time
import dlib
import argparse
import numpy as np
import tensorflow as tf
from model.mobilenetv2 import AgenderNetMobileNetV2
from keras import backend as K 
from utils.stream import WebcamVideoStream
from utils.stream import FPS
from utils.sort import (
    convert_bbox_to_z, convert_x_to_bbox, KalmanBoxTracker, Sort, 
    associate_detections_to_trackers
)

global model, graph
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

def getPosFromRect(rect):
    return (rect.left(), rect.top(), rect.right(), rect.bottom())

def get_result(X):
    with graph.as_default():
        result = model.predict(X)
        return result
def main():
    print("[INFO] sampling frames...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_5_face_landmarks.dat')
    time.sleep(1.0)
    stream = cv2.VideoCapture("data/[MV] Fortune Cookie in Love (Fortune Cookie Yang Mencinta) - JKT48.mp4")
    # stream = WebcamVideoStream("data/[MV] Fortune Cookie in Love (Fortune Cookie Yang Mencinta) - JKT48.mp4").start()
    fps = FPS().start()
    # frame = stream.read()
    start = time.time()
    mot_tracker = Sort()
    while fps._numFrames < args['num_frames']:
        grabbed, frame = stream.read()
        frame = cv2.resize(frame, (640, 360))
        rects = detector(frame, 0)
        if len(rects) > 0 :
            shapes = dlib.full_object_detections()
            for rect in rects:
                shapes.append(predictor(frame, rect))
            faces = dlib.get_face_chips(frame, shapes, size=96, padding=0.4)
            faces = np.array(faces)
            faces = model.prep_image(faces)
            result = get_result(faces)
            genders, ages = model.decode_prediction(result)
            genders = np.where(genders == 0, 'F', 'M')
            for (i, rect) in enumerate(rects):
                (left, top, right, bottom) = getPosFromRect(rect)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "{:.0f}, {}".format(ages[i], genders[i]), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
    
        fps.update()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # print('Frame fetched', stream._numFrames)
    # print('Frame displayed', fps._numFrames)
    print('Elapsed : {:.2f}'.format(time.time() - start))
    stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('[INFO] Load model')
    model = AgenderNetMobileNetV2()
    print('[INFO] Load weight')
    model.load_weights('model/weight/mobilenetv2/model.10-3.8290-0.8965-6.9498.h5')
    graph = tf.get_default_graph()
    main()
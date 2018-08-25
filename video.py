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
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", default=0,
                help="Video stream source, default will be webcam (0)")
args = vars(ap.parse_args())


def get_pos_from_rect(rect):
    return (rect.left(), rect.top(), rect.right(), rect.bottom())


def get_result(X):
    with graph.as_default():
        result = model.predict(X)
        return result


def main():
    print("[INFO] sampling frames...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'model/shape_predictor_5_face_landmarks.dat')
    time.sleep(2.0)
    stream = WebcamVideoStream(args['src']).start()
    fps = FPS().start()
    start = time.time()
    mot_tracker = Sort()
    grabbed, frame = stream.read()
    while grabbed:
        frame = cv2.resize(frame, (1280, 720))
        if fps._numFrames % 3 == 0:
            rects = detector(frame, 0)
        dets = np.array([get_pos_from_rect(rect) for rect in rects])
        ages = np.empty((len(dets)))
        genders = np.empty((len(dets)))
        if len(rects) > 0:
            shapes = dlib.full_object_detections()
            for rect in rects:
                shapes.append(predictor(frame, rect))
            faces = dlib.get_face_chips(frame, shapes, size=96, padding=0.4)
            faces = np.array(faces)
            faces = model.prep_image(faces)
            result = get_result(faces)
            genders, ages = model.decode_prediction(result)

        mot_tracker.update(dets, genders, ages)
        for tracker in mot_tracker.trackers:
            (left, top, right, bottom) = convert_x_to_bbox(
                tracker.kf.x[:4, :]).astype('int').flatten()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            age = tracker.smooth_age()
            gender = 'M' if tracker.smooth_gender() == 1 else 'F'
            cv2.putText(frame, "id: {} {} {}".format(tracker.id, gender, age),
                        (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.putText(frame, "{:.1f} FPS".format(fps.fps()), (1100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()
        grabbed, frame = stream.read()
    fps.stop()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('[INFO] Load model')
    model = AgenderNetMobileNetV2()
    print('[INFO] Load weight')
    model.load_weights(
        'model/weight/mobilenetv2/model.10-3.8290-0.8965-6.9498.h5')
    graph = tf.get_default_graph()
    main()

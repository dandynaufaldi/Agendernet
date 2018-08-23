import os
import cv2
import dlib
import numpy as np


def resize_square_image(image: np.ndarray, size: int =140):
    """
    Resize image and make it square

    Parameters
    ----------
    image   : numpy array -> with dtype uint8 and shape (W, H, 3)
        Image to be resized
    size        : int
        Size of image to be returned

    Returns
    ----------
    resized    : numpy array -> with dtype uint8 and shape (size, size, 3)
        Resized image
    """
    BLACK = [0, 0, 0]
    h = image.shape[0]
    w = image.shape[1]
    if w < h:
        border = h-w
        image = cv2.copyMakeBorder(
            image, 0, 0, border, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        border = w-h
        image = cv2.copyMakeBorder(
            image, border, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized


def align_faces(image: np.ndarray, padding: float=0.4, size: int=140, predictor_path: str=os.path.dirname(
        os.path.dirname(__file__)) + '/model/shape_predictor_5_face_landmarks.dat'):
    """
    Get aligned faces from image if face is detected, else just resize
    Parameters
    ----------
    image   : numpy array -> with dtype uint8 and shape (W, H, 3)
        Image to be processed
    padding : float
        Padding for aligned face
    size    : int
        Size of image to be returned
    path    : string
        Path to dlib facial landmark detector
    Returns
    ----------
    result    : numpy array -> with dtype uint8 and shape (size, size, 3)
        Processed image
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rects = detector(image, 1)
    result = None

    # if detect exactly 1 face, get aligned face
    if len(rects) == 1:
        shape = predictor(image, rects[0])
        result = dlib.get_face_chip(image, shape, padding=padding, size=size)

    # else use resized full image
    else:
        result = resize_square_image(image, size)
    return result

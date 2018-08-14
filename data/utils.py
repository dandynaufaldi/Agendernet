import numpy as np
import cv2
import dlib
import os
import pandas as pd
import datetime
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.io import loadmat


def clean_data(db_frame: pd.DataFrame):
    """
    Clean DataFrame from abnormal data

    Parameters
    ----------
    db_frame   : pandas DataFrame
        DataFrame generated from .mat file to be cleaned

    Returns
    ----------
    cleaned    : pandas DataFrame
        Cleaned DataFrame

    """
    cleaned = db_frame.loc[(db_frame['age'] >= 0) &
                           (db_frame['age'] <= 100) &
                           (~db_frame['face_score'].isnull()) &
                           (db_frame['second_face_score'].isnull()) &
                           (~db_frame['gender'].isnull()),
                           ['db_name', 'full_path', 'age', 'gender']
                           ]
    return cleaned


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


def align_one_face(image: np.ndarray,
                   padding: float =0.4,
                   size: int =140,
                   predictor_path: str='shape_predictor_5_face_landmarks.dat'):
    """
    Get 1 aligned face from image if exist, else just resize
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


def get_year(mat_date):
    """
    Calc year from matlab's date format
    Parameters
    ----------
    mat_date : matlab's datenum
        Date to be converted
    Return
    ----------
    year : int
        Year from matlab datenum
    """
    temp = int(mat_date)
    year = datetime.fromordinal(max(temp - 366, 1)).year
    return year


def load_data(db_name: str, path: str):
    """
    Load data from .mat file (IMDB-Wiki dataset)

    Parameters
    ----------
    db_name : string
        Name of db ['wiki', 'imdb']
    path : string
        path to .mat file

    Returns
    -------
    pandas DataFrame
        Contain db_name, full_path, gender, face_score, seconda_face_score, age
    """

    data = loadmat(path)
    births = data[db_name]['dob'][0, 0][0]
    births = np.array([get_year(birth) for birth in list(births)])
    takens = data[db_name]['photo_taken'][0, 0][0]

    # to save result data
    result = dict()
    col_name = ['full_path', 'gender', 'face_score', 'second_face_score']
    for col in col_name:
        result[col] = data[db_name][col][0, 0][0]
    result['age'] = takens - births
    result['full_path'] = result['full_path'].map(lambda x: x[0])
    # save as pandas dataframe
    col_name.append('age')
    result = pd.DataFrame(data=result, columns=col_name)
    result['db_name'] = db_name

    # handle inf value
    result = result.replace([-np.inf, np.inf], np.nan)
    return result

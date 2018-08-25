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
    # save as pandas dataframe
    col_name.append('age')
    result = pd.DataFrame(data=result, columns=col_name)
    result['full_path'] = result['full_path'].map(lambda x: x[0])
    result['db_name'] = db_name

    # handle inf value
    result = result.replace([-np.inf, np.inf], np.nan)
    return result

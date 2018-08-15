import numpy as np
import cv2
import os
from keras.utils import Sequence, np_utils


def load_image(db: np.ndarray, paths: np.ndarray, size: int):
    """Load image from disk

    Parameters
    ----------
    db : numpy ndarray
        DB's name
    paths : np.ndarray
        Path to imahe
    size : int
        Size of image output

    Returns
    -------
    numpy ndarray
        Array of loaded and processed image
    """

    images = [cv2.imread('data/{}_aligned/{}'.format(db, img_path))
              for (db, img_path) in zip(db, paths)]
    images = [cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) for image in images]
    return np.array(images, dtype='uint8')


class DataGenerator(Sequence):
    """
    Custom data generator inherits Keras Sequence class with multiprocessing support
    Parameters
    ----------
    model : Keras Model
        Model to be used in data preprocessing
    db : np.ndarray
        Array of db name
    paths : np.ndarray
        Array of image paths
    age_label : np.ndarray
        Array of age labels
    gender_label : np.ndarray
        Array of gender label
    batch_size : int
        Size of data generated at once
    """

    def __init__(
            self,
            model, db: np.ndarray,
            paths: np.ndarray,
            age_label: np.ndarray,
            gender_label: np.ndarray,
            batch_size: int):
        self.db = db
        self.paths = paths
        self.age_label = age_label
        self.gender_label = gender_label
        self.batch_size = batch_size
        self.model = model
        self.input_size = model.input_size
        self.categorical = True if model.__class__.__name__ != 'AgenderSSRNet' else False

    def __len__(self):
        return int(np.ceil(len(self.db) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = load_image(db, paths, self.input_size)
        X = self.model.prep_image(batch_x)
        del db, paths, batch_x

        batch_age = self.age_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        age = batch_age
        if self.categorical:
            age = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender_label[idx * self.batch_size:(idx + 1) * self.batch_size]
        gender = batch_gender
        if self.categorical:
            gender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': age,
             'gender_prediction': gender}

        return X, Y

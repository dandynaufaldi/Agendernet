import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import KFold
from model.inceptionv3 import AgenderNetInceptionV3
from model.mobilenetv2 import AgenderNetMobileNetV2
from model.ssrnet import AgenderSSRNet
from utils.generator import DataGenerator
from utils.callback import DecayLearningRate

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2'],
                    default='mobilenetv2',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=50,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='Size of data batch to be used')
parser.add_argument('--num_worker',
                    default=4,
                    type=int,
                    help='Number of worker to process data')


def load_data():
    """
    Load dataset (IMDB, Wiki, Adience)

    Returns
    -------
    db : numpy ndarray
        array of db name
    paths : numpy ndarray
        array of image paths
    age_label : numpy ndarray
        array of age labels
    gender_label : numpy ndarray
        array of gender labels
    """

    wiki = pd.read_csv('data/db/wiki_cleaned.csv')
    imdb = pd.read_csv('data/db/imdb_cleaned.csv')
    adience = pd.read_csv('data/db/adience_cleaned.csv')
    data = pd.concat([wiki, imdb, adience], axis=0)
    del wiki, imdb, adience

    db = data['db_name'].values
    paths = data['full_path'].values
    age_label = data['age'].values.astype('uint8')
    gender_label = data['gender'].values.astype('uint8')
    return db, paths, age_label, gender_label


def mae(y_true, y_pred):
    """Custom MAE for 101 age class, apply softmax regression

    Parameters
    ----------
    y_true : tensor
        ground truth
    y_pred : tensor
        prediction from model

    Returns
    -------
    float
        MAE score
    """

    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    args = parser.parse_args()
    MODEL = args.model
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker

    db, paths, age_label, gender_label = load_data()
    n_fold = 1
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(age_label)
    for train_idx, test_idx in kf_split:
        model = None
        if MODEL == 'ssrnet':
            model = AgenderSSRNet(64, [3, 3, 3], 1.0, 1.0)
        elif MODEL == 'inceptionv3':
            model = AgenderNetInceptionV3()
        else:
            model = AgenderNetMobileNetV2()
        train_db = db[train_idx]
        train_paths = paths[train_idx]
        train_age = age_label[train_idx]
        train_gender = gender_label[train_idx]

        test_db = db[test_idx]
        test_paths = paths[test_idx]
        test_age = age_label[test_idx]
        test_gender = gender_label[test_idx]

        losses = {
            "age_prediction": "categorical_crossentropy",
            "gender_prediction": "categorical_crossentropy",
        }
        metrics = {
            "age_prediction": mae,
            "gender_prediction": "acc",
        }
        if MODEL == 'ssrnet':
            losses = {
                "age_prediction": "mae",
                "gender_prediction": "mae",
            }
            metrics = {
                "age_prediction": "mae",
                "gender_prediction": "binary_accuracy",
            }

        callbacks = [
            ModelCheckpoint(
                "train_weight/{}-{epoch:02d}-{val_loss:.4f}-{val_gender_prediction_acc:.4f}-{val_age_prediction_mae:.4f}.h5".format(
                    MODEL),
                verbose=1, save_best_only=True, save_weights_only=True),
            CSVLogger('train_log/{}-{}.log'.format(MODEL, n_fold))]
        if MODEL == 'ssrnet':
            callbacks = [
                ModelCheckpoint(
                    "train_weight/{}-{epoch:02d}-{val_loss:.4f}-{val_gender_prediction_binary_accuracy:.4f}-{val_age_prediction_mean_absolute_error:.4f}.h5".format(
                        MODEL),
                    verbose=1, save_best_only=True, save_weights_only=True),
                CSVLogger('train_log/{}-{}.log'.format(MODEL, n_fold)),
                DecayLearningRate([30, 60])]
        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.fit_generator(
            DataGenerator(model, train_db, train_paths, train_age, train_gender, BATCH_SIZE),
            validation_data=DataGenerator(model, test_db, test_paths, test_age, test_gender, BATCH_SIZE),
            epochs=EPOCH,
            verbose=2,
            workers=NUM_WORKER,
            use_multiprocessing=True,
            max_queue_size=int(BATCH_SIZE * 2),
            callbacks=callbacks
        )
        n_fold += 1
        del train_db, train_paths, train_age, train_gender
        del test_db, test_paths, test_age, test_gender


if __name__ == '__main__':
    main()

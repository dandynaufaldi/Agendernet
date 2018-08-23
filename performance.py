import argparse
import timeit
import cv2
import logging
import numpy as np
from model.inceptionv3 import AgenderNetInceptionV3
from model.mobilenetv2 import AgenderNetMobileNetV2
from model.ssrnet import AgenderSSRNet

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    required=True,
                    choices=['mobilenetv2', 'inceptionv3', 'ssrnet'],
                    help="model name to be used")


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def predictone(model, x):
    res = model.predict(x)


def proces_time(wrapped):
    number = 100
    elapsed = timeit.repeat(wrapped, repeat=10, number=number)
    elapsed = np.array(elapsed)
    per_pass = elapsed / number
    mean = np.mean(per_pass) * 1000
    std = np.std(per_pass) * 1000
    result = '{:6.2f} msec/pass +- {:6.2f} msec'.format(mean, std)
    return result


def main():
    args = parser.parse_args()
    MODEL = args.model

    model = None
    logger.info('Load model and weight')
    if MODEL == 'mobilenetv2':
        model = AgenderNetMobileNetV2()
        model.load_weights('model/weight/mobilenetv2/model.10-3.8290-0.8965-6.9498.h5')
    elif MODEL == 'inceptionv3':
        model = AgenderNetInceptionV3()
        model.load_weights('model/weight/inceptionv3/model.16-3.7887-0.9004-6.6744.h5')
    else:
        model = AgenderSSRNet(64, [3, 3, 3], 1.0, 1.0)
        model.load_weights('model/weight/ssrnet/model.37-7.3318-0.8643-7.1952.h5')

    logger.info('Read image')
    image = cv2.imread('data/imdb_aligned/02/nm0000002_rm1346607872_1924-9-16_2004.jp')
    image = cv2.resize(image, (model.input_size, model.input_size))
    image = np.expand_dims(image, axis=0)
    image = model.prep_image(image)

    logger.info('Predict with {}'.format(MODEL))
    wrapped = wrapper(predictone, model, image)
    logger.info(proces_time(wrapped))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()

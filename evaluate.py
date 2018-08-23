import numpy as np
import pandas as pd
import argparse
import logging
import time
import cv2
from utils.image import align_one_face, resize_square_image
from tqdm import tqdm
from model.inceptionv3 import AgenderNetInceptionV3
from model.mobilenetv2 import AgenderNetMobileNetV2
from model.ssrnet import AgenderSSRNet

parser = argparse.ArgumentParser()
parser.add_argument('--db_name',
                    required=True,
                    help='name of dataset .csv file in data/db/ folder')
parser.add_argument('--model',
                    required=True,
                    choices=['mobilenetv2', 'inceptionv3', 'ssrnet'],
                    help="model name to be used")


def main():
    args = parser.parse_args()
    DB = args.db_name
    MODEL = args.model

    data = pd.read_csv('data/db/{}.csv'.format(DB))
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
    images = [cv2.imread('{}_aligned/{}'.format(DB, path))
              for path in tqdm(data.full_path.values)]
    images = [cv2.resize(image, (model.input_size, model.input_size))
              for image in images]
    images = np.array(images)
    images = model.prep_image(images)

    logger.info('Predict data')
    start = time.time()
    prediction = model.predict(images)
    pred_gender, pred_age = model.decode_prediction(prediction)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    result = pd.DataFrame()
    result['full_path'] = data['full_path']
    result['age'] = pred_age
    result['gender'] = pred_gender
    result.to_csv('result/{}.csv'.format(DB), index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()

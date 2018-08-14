import os
import cv2
import dlib
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    required=True,
                    choices=['imdb', 'wiki', 'utkface', 'fgnet', 'adience'],
                    help='Dataset name')
parser.add_argument('--num_worker',
                    default=2,
                    type=int,
                    help="num of worker")
args = parser.parse_args()
DATASET = args.dataset
WORKER = args.num_worker
predictor = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')


def align_and_save(path: str):
    """
    Get aligned face and save to disk

    Parameters
    ----------
    path : string
        path to image

    Returns
    -------
    integer
        flag to mark. 1 if success detect face, 0 if fail
    """

    RES_DIR = '{}_aligned'.format(DATASET)
    if os.path.exists(os.path.join(RES_DIR, path)):
        return 1
    flname = os.path.join(DATASET, path)
    image = cv2.imread(flname)
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 0)
    # if detect exactly 1 face, get aligned face
    if len(rects) == 1:
        shape = predictor(image, rects[0])
        result = dlib.get_face_chip(image, shape, padding=0.4, size=140)
        folder = os.path.join(RES_DIR, path.split('/')[0])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        flname = os.path.join(RES_DIR, path)
        if not os.path.exists(flname):
            cv2.imwrite(flname, result)
        return 1
    return 0


def main():
    args = parser.parse_args()
    DATASET = args.dataset
    WORKER = args.num_worker
    data = pd.read_csv('db/{}.csv'.format(DATASET))
    # detector = dlib.get_frontal_face_detector()

    paths = data['full_path'].values
    print('[PREPROC] Run face alignment...')
    with Pool(processes=WORKER) as p:
        res = []
        max_ = len(paths)
        with tqdm(total=max_) as pbar:
            for i, j in tqdm(enumerate(p.imap(align_and_save, paths))):
                pbar.update()
                res.append(j)
        data['flag'] = res

        # create new db with only successfully detected face
        data = data.loc[data['flag'] == 1, list(data)[:-1]]
        data.to_csv('db/{}_cleaned.csv'.format(DATASET), index=False)


if __name__ == '__main__':
    main()

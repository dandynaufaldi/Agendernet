import argparse
import os
import numpy as np
import utils
import pandas as pd
import re
from glob import glob
parser = argparse.ArgumentParser()
parser.add_argument('--db_name',
                    required=True,
                    choices=['imdb', 'wiki', 'utkface', 'fgnet', 'adience'],
                    help='Dataset name')
parser.add_argument('--path',
                    required=True,
                    help="Path to dataset folder")


def make_from_imdb(path: str):
    """Create .csv file as db from IMDB dataset
    Parameters
    ----------
    path : string
        Path to IMDB dataset folder
    """

    data = utils.load_data('imdb', path+'/imdb.mat')
    data = utils.clean_data(data)
    data['db_name'] = path
    data.to_csv('db/imdb.csv',  columns=['db_name', 'full_path', 'age', 'gender'], index=False)


def make_from_wiki(path: str):
    """Create .csv file as db from Wiki dataset
    Parameters
    ----------
    path : string
        Path to Wiki dataset folder
    """

    data = utils.load_data('wiki', path+'/wiki.mat')
    data = utils.clean_data(data)
    data['db_name'] = path
    data.to_csv('db/wiki.csv',  columns=['db_name', 'full_path', 'age', 'gender'], index=False)


def make_from_utkface(path: str):
    """Create .csv file as db from UTKface dataset
    Parameters
    ----------
    path : string
        Path to UTKface dataset folder
    """

    image_list = []
    for i in range(1, 4):
        image_list.extend(glob(os.path.join(path, 'part{}/*.jpg'.format(i))))

    result = dict()
    age = [int(im.split('/')[-1].split('_')[0]) for im in (image_list)]
    gender = [im.split('/')[-1].split('_')[1] for im in (image_list)]
    result['full_path'] = image_list
    result['age'] = age
    result['gender'] = gender

    result = pd.DataFrame.from_dict(result)
    result = result.loc[(result['gender'] != '3') & (result['gender'] != '')]
    result['gender'] = result['gender'].astype('int8') ^ 1

    def removedb(row):
        res = row.split('/')[1:]
        return '/'.join(res)

    result['full_path'] = result['full_path'].map(removedb)
    result['db_name'] = path
    result = result[['db_name', 'full_path', 'age', 'gender']]
    result.to_csv('db/utkface.csv', index=False)


def make_from_fgnet(path: str):
    """Create .csv file as db from FGNET dataset
    Parameters
    ----------
    path : string
        Path to FGNET dataset folder
    """

    pattern = path + '/images/*.JPG'
    paths = glob(pattern)
    data = pd.DataFrame()
    data['full_path'] = paths
    data['db_name'] = path
    p = re.compile('[0-9]+')

    def get_age(row: str):
        flname = row.split('/')[-1]
        age = flname.split('.')[0].split('A')[-1]
        age = p.match(age).group()
        return int(age)
    data['age'] = data['full_path'].map(get_age)

    def clean_path(row: str):
        return '/'.join(row.split('/')[1:])
    data['full_path'] = data['full_path'].map(clean_path)
    data.to_csv('db/fgnet.csv', columns=['db_name', 'full_path', 'age'], index=False)


def make_from_adience(path: str):
    """Create .csv file as db from Adience dataset
    Parameters
    ----------
    path : string
        Path to Adience dataset folder
    """

    fold_files = glob(path + "/*.txt")
    data = pd.read_csv(fold_files[0], sep='\t')
    for file in fold_files[1:]:
        temp = pd.read_csv(file, sep='\t')
        data = pd.concat([data, temp])
    data['full_path'] = data['user_id'] + '/coarse_tilt_aligned_face.' + \
        data['face_id'].astype('str') + '.' + data['original_image']

    def rnd(low: int, high: int):
        return np.random.randint(low, high + 1)

    def makeAge(age: int):
        if age == '(0, 2)':
            return rnd(0, 2)
        elif age == '(4, 6)':
            return rnd(4, 6)
        elif age in ['(8, 12)', '(8, 23)']:
            return rnd(8, 12)
        elif age == '(15, 20)':
            return rnd(15, 20)
        elif age in ['(25, 32)', '(27, 32)']:
            return rnd(25, 32)
        elif age in ['(38, 43)', '(38, 48)', '(38, 42)']:
            return rnd(38, 43)
        elif age == '(48, 53)':
            return rnd(48, 53)
        elif age == '(60, 100)':
            return rnd(60, 100)
        elif age == 'None':
            return np.nan
        else:
            return int(age)
    data['age'] = data['age'].map(makeAge)
    data['db_name'] = 'adience'
    data = data.loc[(~data['age'].isnull()) & ((data['gender'] == 'f') | (
        data['gender'] == 'm')), ['db_name', 'full_path', 'age', 'gender']]
    gender = {'m': 1, 'f': 0}
    data['gender'] = data['gender'].map(gender)
    data.to_csv('db/adience.csv', index=False)


def main():
    args = parser.parse_args()
    DB = args.db_name
    PATH = args.path
    command = {'imdb': make_from_imdb,
               'wiki': make_from_wiki,
               'utkface': make_from_utkface,
               'fgnet': make_from_fgnet,
               'adience': make_from_adience}
    command[DB](PATH)


if __name__ == '__main__':
    main()

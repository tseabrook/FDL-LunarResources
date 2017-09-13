#Written by Yarin Gal
#yg279@cam.ac.uk

import numpy as np
import glob, os, cv2
from PIL import Image

# load the data, assuming folder structure is [base folder]/[annotator]/*.png with multiple annotators.
def load_data(base_folder, start_row=0, end_row=np.inf, start_col=0, end_col=np.inf, is_augment=True):
    file_names = glob.glob(base_folder + '/*.png') + glob.glob(base_folder + '/*/*.png')
    file_names += glob.glob(base_folder + '/*.tif') + glob.glob(base_folder + '/*/*.tif')
    arrs = []
    file_names = filter_row_col(file_names, start_row, end_row, start_col, end_col)
    for name in file_names:
        x = cv2.resize(cv2.imread(name, 0), (32, 32)).astype(np.float32)  # Load an color image in grayscale
        if is_augment:
            flip_v = cv2.flip(x, 0)
            flip_h = cv2.flip(x, 1)
            rotate_180 = cv2.flip(flip_h, 0)
            arrs += [x, flip_v, flip_h, rotate_180]
        else:
            arrs += [x]
    X = np.array(arrs).reshape((-1, 32*32)) / 255.
    print('file_names len', len(file_names), 'X shape', X.shape)
    return X, file_names

# filter files only within row/column range
def filter_row_col(file_names, start_row, end_row, start_col, end_col):
    out = []
    for name in file_names:
        base_name = os.path.basename(name)
        if '_d' in base_name and '_x' in base_name and '_y' in base_name:
            col, row = map(int, base_name.split('_x')[1].split('.')[0].split('_y'))
        else:
            col, row = map(int, base_name.split('_')[1].split('.')[0].split('-'))
        if start_row <= row and row < end_row and start_col <= col and col < end_col:
            out += [name]
    return out

# create a dataset from loaded pos / neg arrays
def create_dataset(pos_arrs, neg_arrs):
    X = np.concatenate([pos_arrs, neg_arrs])
    # y = np.identity(2)[pos_arrs.shape[0] * [1] + neg_arrs.shape[0] * [0]]
    y = np.concatenate([np.ones((pos_arrs.shape[0])), np.zeros((neg_arrs.shape[0]))]).reshape((-1, 1))
    print('X shape', X.shape, 'y shape', y.shape)
    return X, y

# shuffle a dataset
def shuffle(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

# split X, y to train/test sets; note that the test split ignores overlapping tiles
def split(X, y, train_frac=0.7):
    nb_train = int(train_frac * X.shape[0])
    return (X[:nb_train], y[:nb_train]), (X[nb_train:], y[nb_train:])

# split to train/test sets; note that the test split ignores overlapping tiles
def split(X, train_frac=0.7):
    nb_train = int(train_frac * X.shape[0])
    return X[:nb_train], X[nb_train:]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io
from sklearn.metrics import fbeta_score

import pathfinder
import utils

rng = np.random.RandomState(37145)


def read_compressed_image(dataset, idx):
    if dataset == 'train':
        prefix = 'train-compressed/train_'
    elif dataset == 'test':
        prefix = 'test-compressed/test_'
    else:
        raise
    path = pathfinder.DATA_PATH + prefix + str(idx) + '.npz'
    image = utils.load_np(path)
    if 'arr' not in image:
        print path
    return image['arr']

def read_image(dataset, idx):
    if dataset == 'train':
        prefix = 'train-tif-v2/train_'
    elif dataset == 'test':
        prefix = 'test-tif-v2/test_'
    else:
        raise
    path = pathfinder.DATA_PATH + prefix + str(idx) + '.tif'
    image = io.imread(path)
    image = np.swapaxes(image,0,2)
    return image

def save_image_compressed(dataset, idx):
    np_image = read_image(dataset, idx)
    if dataset == 'train':
        prefix = 'train-compressed/train_'
    elif dataset == 'test':
        prefix = 'test-compressed/test_'
    else:
        raise
    path = pathfinder.DATA_PATH + prefix + str(idx) + '.npz'
    utils.savez_compressed_np(np_image, path)
    

def get_labels():
    df = pd.read_csv(pathfinder.DATA_PATH+'train.csv')
    df = pd.concat([df['image_name'], df.tags.str.get_dummies(sep=' ')], axis=1)
    cols = list(df.columns.values) #Make a list of all of the columns in the df
    weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
    rare_labels = ['slash_burn','conventional_mine', 'bare_ground', 'artisinal_mine',
                    'blooming','selective_logging','blow_down']
    for label in weather_labels:
        cols.pop(cols.index(label)) #Remove b from list
    for label in rare_labels:
        cols.pop(cols.index(label)) #Remove b from list
    df = df[weather_labels+rare_labels+cols] #
    return df

def get_labels_array():
    df = get_labels()
    only_labels = df.drop(['image_name'], axis = 1, inplace = False)
    only_labels = only_labels.as_matrix()
    return only_labels

def get_d_labels():
    d_labels = {}
    df = get_labels()
    label_array = get_labels_array()
    for index, row in df.iterrows():
        d_labels[row['image_name']] = label_array[index]
    return d_labels

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def top_occ(feat_comb, n_top = 5):
    # a method for printing top occurences of feature combinations
    # built for checking if split is stratified
    n_total_samples = len(feat_comb)
    feat_comb_occ = np.bincount(feat_comb)
    top = feat_comb_occ.argsort()[-n_top:][::-1]
    for idx, fc in enumerate(top):
        print idx, fc, 1.0*feat_comb_occ[fc]/n_total_samples
    print 'checksum', sum(feat_comb)

def make_stratified_split(no_folds=5, verbose=False):
    df = get_labels()
    only_labels = df.drop(['image_name'], axis = 1, inplace = False)
    only_labels = only_labels.as_matrix()
    if verbose: print 'labels shape', only_labels.shape
    feat_comb = only_labels.dot(1 << np.arange(only_labels.shape[-1] - 1, -1, -1))
    feat_comb_set = set(feat_comb)
    feat_comb_occ = np.bincount(feat_comb)
    feat_comb_high = np.where(feat_comb_occ >= no_folds)[0]
    n_total_samples = 0
    folds = [[] for _ in range(no_folds)]
    for fc in feat_comb_high:
        idcs = np.where(feat_comb == fc)[0]
        chunks = chunkIt(idcs,no_folds)
        # print len(idcs), [len(chunk) for chunk in chunks]
        rng.shuffle(chunks)
        for idx, chunk in enumerate(chunks):
            folds[idx].extend(chunk)

    feat_comb_low = np.where(np.logical_and(feat_comb_occ < no_folds, feat_comb_occ > 0))[0]
    low_idcs = []
    for fc in feat_comb_low:
        idcs = np.where(feat_comb == fc)[0]
        low_idcs.extend(idcs)

    chunks = chunkIt(low_idcs,no_folds)
    rng.shuffle(chunks)
    for idx, chunk in enumerate(chunks):
        folds[idx].extend(chunk)


    n_samples_fold = 0
    for f in folds:
        n_samples_fold += len(f)

    if verbose:
        print 'n_samples_fold', n_samples_fold
        top_occ(feat_comb)
        for f in folds:
            top_occ(feat_comb[f])

    return folds

def generate_compressed(img_ids):
    for idx, img_id in enumerate(img_ids):
        if idx%100 == 0:
            print idx, '/', len(img_ids)
        save_image_compressed('train', img_id)

def generate_compressed_trainset():
    folds = make_stratified_split(no_folds=5)
    all_ids = folds[0] + folds[1] + folds[2] + folds[3] +folds[4]
    bad_ids = [18772, 28173, 5023]
    img_ids = [x for x in all_ids if x not in bad_ids]
    generate_compressed(img_ids)

def f2_score(y_true, y_pred, average='samples'):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average=average)


if __name__ == "__main__":
    #make_stratified_split()
    bad_ids = [18772, 28173, 5023]
    generate_compressed(bad_ids)
    #generate_compressed_trainset()
    #get_labels()





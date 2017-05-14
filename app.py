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
    only_labels = get_labels_array()
    # df = get_labels()
    # only_labels = df.drop(['image_name'], axis = 1, inplace = False)
    # only_labels = only_labels.as_matrix()
    # if verbose: print 'labels shape', only_labels.shape
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


def investigate_labels():
    only_labels_all = get_labels_array()
    feat_comb = only_labels_all.dot(1 << np.arange(only_labels_all.shape[-1] - 1, -1, -1))
    feat_comb_set = set(feat_comb)
    print 'number of combinations when all labels are present'
    print len(feat_comb_set)
    feat_comb_occ = np.bincount(feat_comb)

    for i in range(17):
        print 'cutting out label', i
        only_labels_sel = np.copy(only_labels_all)
        only_labels_sel = np.delete(only_labels_sel,i, axis=1)
        print only_labels_sel.shape
        feat_comb = only_labels_sel.dot(1 << np.arange(only_labels_sel.shape[-1] - 1, -1, -1))
        feat_comb_set = set(feat_comb)
        print 'number of features', len(feat_comb_set)
        feat_comb_occ = np.bincount(feat_comb)

def get_pos_neg_ids(label_id):
    labels = get_labels_array()
    pos_ids = np.where(labels[:,label_id])
    neg_ids = np.where(labels[:,label_id]==0)
    return pos_ids[0], neg_ids[0]


def get_biases():
    df = get_labels()
    df = df.drop(['image_name'], axis = 1, inplace = False)
    label_list = list(df.columns.values)
    histo = df[label_list].sum()
    biases =  1.*np.int32(histo)/len(df.index)
    return biases


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

def _test_get_pos_neg_ids():
    for i in range(17):
        pos_ids, neg_ids = get_pos_neg_ids(i)
        print i, len(pos_ids), len(neg_ids), len(pos_ids)+len(neg_ids)
        print get_labels_array()[pos_ids[0]]
        print get_labels_array()[pos_ids[1]]
        print get_labels_array()[neg_ids[0]]
        print get_labels_array()[neg_ids[1]]
        print 'test done'


def logloss(predictions, targets, epsilon=1.e-7, skewing_factor = 1.):
    preds = np.clip(predictions, epsilon, 1.-epsilon)
    weighted_bce = - skewing_factor * targets * np.log(preds) - (1-targets)*np.log(1-preds)    
    return np.mean(weighted_bce)


if __name__ == "__main__":
    #investigate_labels()
    # bad_ids = [18772, 28173, 5023]
    # generate_compressed(bad_ids)
    # labels = get_labels_array()
    # print labels[:10]
    # print labels.shape

    # a = np.random.randint(0, 2, 1000)
    # b = np.random.randint(0, 2, 1000)
    # f2 = fbeta_score(a, b, beta=2, average=None)
    # print f2
    # tp = sum(a*b)
    # fp = sum(a*(1-b))
    # fn = sum((1-a)*b)
    # print 5.*tp/(5.*tp+4.*fn+fp)

    print 'main function'
    targets = np.array([1,1,0,0,0,1,0,1,1,0,1,0,1])
    predictions = np.array([.8,.95,.1,.05,0,.9,0,.99,.92,.12,.96,.03,.89])
    print skewed_logloss(predictions,targets)
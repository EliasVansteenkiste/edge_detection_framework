import numpy as np
from PIL import Image
import os
import scipy

from sklearn.metrics import fbeta_score

import pathfinder


rng = np.random.RandomState(37145)


# def read_mat(dataset, idx, plot=False):
#     path = pathfinder.DATA_PATH + '/' + dataset + '/' + str(idx) + '.mat'
#     d_mat = io.loadmat(path)
#     print(d_mat['groundTruth'].shape)
#     fmaps = []
#     for midx, gmap in enumerate(d_mat['groundTruth'][0]):
#         if plot:
#             utils_plots.show_img(gmap[0][0][0], dataset+'_'+str(idx)+'_'+str(midx))
#         fmaps.append(gmap[0][0][0])
#     return np.stack(fmaps)



def read_image(dataset, filename):
    path = pathfinder.DATA_PATH + '/' + dataset + '/' + filename
    im = Image.open(path)
    arr = np.asanyarray(im)
    return arr


def read_image_from_path(path):
    return Image.open(path)


def read_image_from_id(id):
    path = pahfinder.DATA_PATH + '/' + id
    im = Image.open(path)
    arr = np.asanyarray(im)
    return arr


def save_image(img_arr, filename, mode='L'):
    scipy.misc.imsave(filename, img_arr)

    # im = Image.fromarray(img_arr, mode=mode)
    # print(im.mode)
    # if im.mode != 'RGB':
    #     im = im.convert('RGB')
    # im.save(filename)


def get_id_pairs(dataset_img, dataset_edges):
    img_path = os.path.join(os.sep, pathfinder.DATA_PATH, dataset_img)
    edges_path = os.path.join(os.sep, pathfinder.DATA_PATH, dataset_edges)

    filenames = os.listdir(img_path)
    id_pairs = []
    for filename in filenames:
        if filename[0] != '.':
            img_filename = os.path.join(os.sep, img_path, filename)
            edges_filename = os.path.join(os.sep, edges_path, filename)
            assert os.path.exists(edges_filename), 'corresponding edge map not found: '+edges_filename
            id_pairs.append((img_filename, edges_filename))
    return id_pairs


def make_splits(lids, fractions):
    fractions = np.array(fractions)
    assert abs(1.-sum(fractions)) < 1e-6
    n = len(lids)
    ns_per_part = np.floor(n * fractions).astype(int)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    splits = []
    ptr = 0
    for nel_idx, nel in enumerate(ns_per_part):
        split = []
        split = [lids[i] for i in idxs[ptr:ptr+nel]]
        splits.append(split)
        ptr += nel
    return splits


def train_val_test_split(id_lists, train_fraction, val_fraction, test_fraction):
    train_ids = []
    val_ids = []
    test_ids = []

    for dataset_idx, id_list in enumerate(id_lists):
        print('dataset', dataset_idx, 'contains', len(id_lists), 'items.')
        train, val, test = make_splits(id_list, [train_fraction, val_fraction, test_fraction])
        train_ids += train
        val_ids += val
        test_ids += test
        print('train_ids', len(train_ids), 'val_ids', len(val_ids), 'test_ids', len(test_ids))

    return {'train': train_ids, 'valid': val_ids, 'test': test_ids}


def f2_score(y_pred, y_true, average='samples'):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average=average)


def f2_score_arr( y_pred, y_true, treshold=.5, average='samples'):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert(len(y_pred.shape)==2)
    assert(len(y_true.shape)==2)
    assert(y_pred.shape[0]==y_true.shape[0])
    n_samples = y_true.shape[0]
    y_pred_cutoff = np.digitize(y_pred, [-0.01,treshold,1.01])-1
    return f2_score(y_true, y_pred_cutoff, average)

def cont_f_score(y_pred, y_true, beta=1.0):
    f_scores = []
    tps = []
    for ipred, itrue in zip(y_pred, y_true):
        ipred = np.array(ipred)
        itrue = np.array(itrue)

        tp = np.sum(itrue * ipred)
        fp = np.sum((1-itrue) * ipred)
        fn = np.sum(itrue * (1-ipred))

        tps.append(tp)

        # print('tp', np.sum(tp), 'fp', np.sum(fp), 'fn', np.sum(fn), 'itrue', np.sum(itrue), 'ipred', np.sum(ipred))

        f_score = (1+beta**2) * tp / ((1+beta**2) * tp + beta**2 * fn + fp + 1.)
        f_scores.append(f_score)

    f_scores = np.array(f_scores)
    mean_f_score = np.mean(f_scores)

    return mean_f_score



if __name__ == "__main__":
    # read_image('test_data', 'test1/trainA/20170831-22-46-33_0000000004.jpg')
    # read_image('test_data', 'test1_hed/trainA/20170831-22-46-33_0000000004.jpg')
    # print(get_id_pairs('test_data/test1/trainA', 'test_data/test1_hed/trainA'))

    dataset1 = app.get_id_pairs('test_data/test1/trainA', 'test_data/test1_hed/trainA')
    img_id_pairs = [dataset1]

    id_pairs = app.train_val_test_split(img_id_pairs, [.5, .25, .25])

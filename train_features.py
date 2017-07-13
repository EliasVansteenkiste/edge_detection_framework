#! /usr/bin/python
import numpy as np
import xgboost as xgb
import os
from sklearn import preprocessing
from sklearn.metrics import fbeta_score

import utils
import app
import cPickle
import pathfinder

seed = 31145
rng = np.random.RandomState(141289)



def learn_bin_class(train_ids, valid_ids, features, f_idx, labels):
    train_X = features[train_ids]
    valid_X = features[valid_ids]

    train_Y = labels[train_ids, f_idx]
    valid_Y = labels[valid_ids, f_idx]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['eta'] = 0.05
    param['max_depth'] = 2
    param['n_estimators'] = 100
    #param['learning_rate'] = 0.1
    param['min_child_weight'] = 1
    param['alpha'] = 1  # L1 regularization term on weights, default 0
    param['lambda'] = 5  # L2 regularization term on weights
    param['lambda_bias'] = 0  # L2 regularization term on bias, default 0
    param['gamma'] = 0.5
    param['subsample']= 0.3
    param['colsample_bytree'] = 0.95
    param['scale_pos_weight'] = 1
    param['silent'] = 1
    param['nthread'] = 10
    param['num_class'] = 1

    watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]
    num_round = 30

    # do the same thing again, but output probabilities
    bst = xgb.train(param, xg_train, num_round, watchlist);
    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob_train = bst.predict(xg_train)
    yprob_valid = bst.predict(xg_valid)
    # print yprob.shape
    # ylabel = yprob > 0.5

    # print 'skewed bce', app.logloss(yprob, valid_Y, skewing_factor=5.)
    #
    # print ('predicting, classification error=%f' % (
    # sum(int(ylabel[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y))))

    return yprob_train, yprob_valid, bst


def build_joint_feature_vector(config_names):
    id_pred = {}
    id_target = {}

    for file_name in config_names:

        if file_name[0:3]=="f95":

            path_valid1 = "/data/plnt/model-predictions/fgodin/" + file_name + "/" + file_name + "_valid_tta_predictions.p"

        else:
            path_valid1 = "/data/plnt/model-predictions/fgodin/" + file_name + "/" + file_name + "_1_predictions.p"

        file = open(path_valid1, mode="rb")
        preds, targets, ids = cPickle.load(file)
        file.close()

        for i in range(preds.shape[0]):

            if ids[i] in id_pred:
                id_pred[ids[i]].append(preds[i])
            else:
                id_pred[ids[i]] = [preds[i]]
                id_target[ids[i]] = np.asarray(targets[i],dtype=np.int)

    for k, v in id_pred.items():
        id_pred[k] = np.concatenate(v)

    return id_pred, id_target


def make_stratified_split(only_labels, no_folds=5, verbose=False, version=1):


    feat_comb = only_labels.dot(1 << np.arange(only_labels.shape[-1] - 1, -1, -1))

    feat_comb_occ = np.bincount(feat_comb)
    feat_comb_high = np.where(feat_comb_occ >= no_folds)[0]

    folds = [[] for _ in range(no_folds)]
    for fc in feat_comb_high:
        idcs = np.where(feat_comb == fc)[0]
        chunks = app.chunkIt(idcs, no_folds)
        # print len(idcs), [len(chunk) for chunk in chunks]
        rng.shuffle(chunks)
        for idx, chunk in enumerate(chunks):
            folds[idx].extend(chunk)

    feat_comb_low = np.where(np.logical_and(feat_comb_occ < no_folds, feat_comb_occ > 0))[0]
    low_idcs = []
    for fc in feat_comb_low:
        idcs = np.where(feat_comb == fc)[0]
        low_idcs.extend(idcs)

    chunks = app.chunkIt(low_idcs, no_folds)
    rng.shuffle(chunks)
    for idx, chunk in enumerate(chunks):
        folds[idx].extend(chunk)

    n_samples_fold = 0
    for f in folds:
        n_samples_fold += len(f)

    if verbose:
        print 'n_samples_fold', n_samples_fold
        app.top_occ(feat_comb)
        for f in folds:
            app.top_occ(feat_comb[f])

    return folds

config_names = [
    "f87_pt-20170623-114248-best",
    #                 "f87-0_pt-20170625-085603-best",
    #                 "f87-1_pt-20170625-085758-best",
    #                 "f87-2_pt-20170625-091432-best",
    #                 "f87-3_pt-20170625-083440-best",
    #
    "f92_pt-20170623-114700-best",
    #                 "f92-0_pt-20170625-020639-best",
    #                 "f92-1_pt-20170625-021713-best",

    #                 "f92-2_pt-20170625-023812-best",
    #                 "f92-3_pt-20170625-020307-best",
    "f95_pt-20170624-035637-best",
    # "f95-0_pt-20170709-154037-best",
    # "f95-1_pt-20170709-214044-best",
    # "f95-2_pt-20170709-131550-best",
    # "f95-3_pt-20170709-184153-best",
    #
    "f97_pt-20170624-002331-best",
    #                 "f97-0_pt-20170624-180609-best",
    #                 "f97-1_pt-20170624-180713-best",
    #                 "f97-2_pt-20170624-180825-best",
    #                 "f97-3_pt-20170624-180932-best"
    #                     "f101_pt-20170626-090424-best",
    #                     "f101-0_pt-20170705-102142-best",
    #                     "f101-1_pt-20170705-165441-best",
    #                     "f101-2_pt-20170705-102109-best",
    #                     "f101-3_pt-20170705-165037-best"
   "f113_pt-20170704-183933-best"
]

x_dict, y_dict = build_joint_feature_vector(config_names)

# just to be sure
x = np.empty((len(x_dict),17*len(config_names)))
y = np.empty((len(x_dict),17),dtype=np.int)

ids = x_dict.keys()
for i in range(len(ids)):
    x[i] = x_dict[ids[i]]
    y[i] = y_dict[ids[i]]


# split deep learning validation set in 2 new sets for training xgboost
folds = make_stratified_split(y,no_folds=3)


threshold = 0.24

sum_f2_scores_train = []
sum_f2_scores_valid = []

models = [[] for i in range(len(folds))]


for fold_id in range(len(folds)):
    train_ids = []
    for i in range(len(folds)):
        if i != fold_id:
            train_ids.extend(folds[i])
    valid_ids = folds[fold_id]
    train_preds = np.empty((len(train_ids),y.shape[1]),dtype=np.float32)
    valid_preds = np.empty((len(valid_ids),y.shape[1]),dtype=np.float32)


    for f_idx in range(0, 17):
        print 'f_idx', f_idx
        train_preds[:, f_idx], valid_preds[:,f_idx], model = learn_bin_class(train_ids, valid_ids, x, f_idx, y)
        models[fold_id].append(model)

    sum_f2_scores_train.append(fbeta_score(y[train_ids], train_preds > threshold, beta=2, average='samples'))
    sum_f2_scores_valid.append(fbeta_score(y[valid_ids], valid_preds > threshold, beta=2, average='samples'))

print "F2 train: ", sum_f2_scores_train, np.mean(sum_f2_scores_train)
print "F2 valid: ", sum_f2_scores_valid, np.mean(sum_f2_scores_valid)

# filename = "xgboost-allfold-"+"-".join(config_names)+".pkl"
# metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
#
# file = open(os.path.join(metadata_dir,filename),"wb")
# cPickle.dump([models,threshold],file)
# file.close()



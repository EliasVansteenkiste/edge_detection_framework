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


def load_features(path):
    img_ids = []
    filelist = os.listdir(path)
    for f in filelist:
        img_id = int(f.split('.')[0])
        img_ids.append(img_id)
    max_id = max(img_ids)

    test_features = utils.load_np(path + '/' + filelist[0])['features']
    features = np.zeros((max_id + 1,) + test_features.shape)
    for f in filelist:
        img_id = int(f.split('.')[0])
        file = utils.load_np(features_path + '/' + f)
        features[img_id] = file['features']
    return features


def learn_weather_class(train_ids, valid_ids, features, labels):
    train_X = features[train_ids]
    valid_X = features[valid_ids]

    weather_labels = labels[:, :4]
    print weather_labels[:10]
    weather_class = np.argmax(weather_labels, axis=1)
    print weather_class[:10]

    train_Y = weather_class[train_ids]
    valid_Y = weather_class[valid_ids]
    y_valid_true = weather_labels[valid_ids]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 8
    param['n_estimators'] = 2000
    param['learning_rate'] = 0.1
    param['min_child_weight'] = 1
    param['gamma'] = 0
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['scale_pos_weight'] = 1
    # param['silent'] = 1
    param['nthread'] = 10
    param['num_class'] = 4

    watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]
    num_round = 60
    # bst = xgb.train(param, xg_train, num_round, watchlist );
    # # get prediction
    # pred = bst.predict( xg_valid );
    # print ('predicting, classification error=%f' % (sum( int(pred[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y)) ))

    # do the same thing again, but output probabilities
    param['objective'] = 'multi:softprob'
    bst = xgb.train(param, xg_train, num_round, watchlist);
    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob = bst.predict(xg_valid).reshape(valid_Y.shape[0], 4)
    ylabel = np.argmax(yprob, axis=1)

    print 'f2_score', app.f2_score_arr(y_valid_true, yprob)
    y_argm = np.argmax(yprob, axis=1)
    y_max_pred = np.zeros_like(yprob)
    y_max_pred[np.arange(y_argm.shape[0]), y_argm] = 1
    print 'f2_score', app.f2_score_arr(y_valid_true, y_max_pred)
    print ('predicting, classification error=%f' % (
    sum(int(ylabel[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y))))


def learn_bin_class(train_ids, valid_ids, features, f_idx, labels, augmentations=False):

    if augmentations:
        train_X = np.vstack([features[i][train_ids] for i in range(len(features))])
        train_Y = np.concatenate([labels[train_ids, f_idx] for _ in range(len(features))])

        if valid_ids is not None:
            valid_X = np.vstack([features[i][valid_ids] for i in range(len(features))])
            valid_Y = np.concatenate([labels[valid_ids, f_idx] for _ in range(len(features))])

    else:

        train_X = features[train_ids]
        train_Y = labels[train_ids, f_idx]
        if valid_ids is not None:
            valid_X = features[valid_ids]
            valid_Y = labels[valid_ids, f_idx]

    xg_train = xgb.DMatrix(train_X, label=train_Y)

    if valid_ids is not None:
        xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['eta'] = 0.05
    param['max_depth'] = 4
    param['n_estimators'] = 100
    #param['learning_rate'] = 0.1
    param['min_child_weight'] = 1
    param['alpha'] = 1  # L1 regularization term on weights, default 0
    param['lambda'] = 10  # L2 regularization term on weights
    param['lambda_bias'] = 0  # L2 regularization term on bias, default 0
    param['gamma'] = 1
    param['subsample']= 0.3
    param['colsample_bytree'] = 0.5
    param['scale_pos_weight'] = 1
    param['silent'] = 1
    param['nthread'] = 10
    param['num_class'] = 1

    if valid_ids is not None:
        watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]
    else:
        watchlist = [(xg_train, 'train')]
    num_round = 30

    # do the same thing again, but output probabilities
    bst = xgb.train(param, xg_train, num_round, watchlist);
    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob_train = bst.predict(xg_train)

    if valid_ids is not None:
        yprob_valid = bst.predict(xg_valid)
    # print yprob.shape
    # ylabel = yprob > 0.5

    # print 'skewed bce', app.logloss(yprob, valid_Y, skewing_factor=5.)
    #
    # print ('predicting, classification error=%f' % (
    # sum(int(ylabel[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y))))

    if valid_ids is not None:
        return yprob_train, yprob_valid, bst
    else:
        return yprob_train, bst


def build_joint_prob_vector(config_names):
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

def build_joint_feature_vector(config_names,ids):

    labels = app.get_labels_array()

    targets = dict(zip(ids,labels[ids]))

    print(len(ids))

    vectors = {}

    error_ids = []

    for valid_id in ids:
        augmentations = []
        for aug in range(1):
            predictions = []
            error = False

            for config_name in config_names:
                try:
                    file = open(os.path.join("/data/plnt/model-predictions/fgodin/",
                                             config_name,"features",
                                             str(valid_id)+"_"+str(aug)+".npy"),"rb")
                    predictions.append(np.load(file))
                    file.close()
                except IOError as e:
                    #print(valid_id)
                    error = True
                    error_ids.append(valid_id)

            #print(np.concatenate(predictions).shape)
            if not error:
                augmentations.append(np.concatenate(predictions))

        vectors[valid_id] = augmentations

    for id in list(set(error_ids)):
        print(str(id)+",")

    return vectors,targets



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

num_classes = 17
def calculate_thresholds(preds,targets):


    num_iters = 50
    best_thresholds = [0.5] * num_classes
    best_scores = [0] * num_classes

    for t in range(num_classes):

        thresholds = [0.5] * num_classes
        for i in range(num_iters):
            ths = [0.5 - (i / float(num_iters)) * 0.5, 0.5 + (i / float(num_iters)) * 0.5]

            for th in ths:
                thresholds[t] = th
                f2 = fbeta_score(targets, preds > thresholds, beta=2, average='samples')
                if f2 > best_scores[t]:
                    best_scores[t] = f2
                    best_thresholds[t] = th

    #print("Best F2: "+str(fbeta_score(targets, preds > best_thresholds, beta=2, average='samples')))

    return best_thresholds

config_names = [
    "f87_pt-20170712-100723-best",
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
    "f97_pt-20170629-104043-best",
    #                 "f97-0_pt-20170624-180609-best",
    #                 "f97-1_pt-20170624-180713-best",
    #                 "f97-2_pt-20170624-180825-best",
    #                 "f97-3_pt-20170624-180932-best"
    #"f101_pt-20170626-090424-best",
    #                     "f101-0_pt-20170705-102142-best",
    #                     "f101-1_pt-20170705-165441-best",
    #                     "f101-2_pt-20170705-102109-best",
    #                     "f101-3_pt-20170705-165037-best"
   "f113_pt-20170704-183933-best"
]

folds = app.make_stratified_split(no_folds=5)
all_ids = folds[0] + folds[1] + folds[2] + folds[3] + folds[4]
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]

x_dict, y_dict = build_joint_feature_vector(config_names,train_ids)

# just to be sure

y = np.empty((len(x_dict),17),dtype=np.int)


i = 0
augmentations = [np.empty((len(x_dict),7808)) for _ in range(8)]

for key, vectors in x_dict.items():

    for j,vector in enumerate(vectors):
        augmentations[j][i] = vector


    y[i] = y_dict[key]

    i+=1


if True:

    threshold = 0.24

    sum_f2_scores_train = []
    sum_f2_scores_valid = []

    sum_f2_scores_train_adaptive_thresholds = []
    sum_f2_scores_valid_adaptive_thresholds = []

    models = [[] for i in range(len(folds))]


    train_preds = np.empty((len(train_ids)*8,y.shape[1]),dtype=np.float32)
    valid_preds = np.empty((len(valid_ids)*8,y.shape[1]),dtype=np.float32)
    train_targets = np.vstack([y[train_ids] for _ in range(8)])
    valid_targets = np.vstack([y[valid_ids] for _ in range(8)])

    for f_idx in range(0, 17):
        print 'f_idx', f_idx
        train_preds[:, f_idx], valid_preds[:,f_idx], model = learn_bin_class(train_ids, valid_ids,augmentations, f_idx, y,augmentations=True)
        #models[fold_id].append(model)

    sum_f2_scores_train.append(fbeta_score(train_targets, train_preds > threshold, beta=2, average='samples'))
    sum_f2_scores_valid.append(fbeta_score(valid_targets, valid_preds > threshold, beta=2, average='samples'))

    print("F2 train: "+str(sum_f2_scores_train[-1]))
    print("F2 valid: " + str(sum_f2_scores_valid[-1]))

        # best_threshold = calculate_thresholds(train_preds,train_targets)
        # sum_f2_scores_train_adaptive_thresholds.append(fbeta_score(train_targets, train_preds > best_threshold, beta=2, average='samples'))
        # sum_f2_scores_valid_adaptive_thresholds.append(fbeta_score(valid_targets, valid_preds > best_threshold, beta=2, average='samples'))

    print "F2 train: ", sum_f2_scores_train, np.mean(sum_f2_scores_train)
    print "F2 valid: ", sum_f2_scores_valid, np.mean(sum_f2_scores_valid)

else:

        train_ids = range(len(y_dict))
        train_preds = np.empty((len(train_ids) * 8, y.shape[1]), dtype=np.float32)
        train_targets = np.vstack([y[train_ids] for _ in range(8)])

        for f_idx in range(0, 17):
            print 'f_idx', f_idx
            train_preds[:, f_idx], model = learn_bin_class(train_ids, None, augmentations, f_idx,
                                                                           y, augmentations=True)

            filename = "xgboost-features-"+str(f_idx)+"-"+"-".join(config_names)+".pkl"
            metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

            model.save_model(os.path.join(metadata_dir,filename))




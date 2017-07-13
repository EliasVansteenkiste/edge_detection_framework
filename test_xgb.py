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

def build_joint_feature_vector(config_names):
    id_pred = {}

    for config_name in config_names:

        predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
        outputs_path = predictions_dir + '/' + config_name
        output_pickle_file = outputs_path + '/%s-%s.pkl' % (config_name, "test_tta")
        preds = cPickle.load(open(output_pickle_file, "rb"))

        ids = preds.keys()

        for i in range(len(preds)):

            if ids[i] in id_pred:
                id_pred[ids[i]].append(preds[ids[i]])
            else:
                id_pred[ids[i]] = [preds[ids[i]]]


    for k, v in id_pred.items():
        id_pred[k] = np.concatenate(v)

    return id_pred


config_names = [
    "f87_pt-20170623-114248-best",
    #                 "f87-0_pt-20170625-085603-best",
     #                "f87-1_pt-20170625-085758-best",
    #                 "f87-2_pt-20170625-091432-best",
    #                 "f87-3_pt-20170625-083440-best",
    #
    "f92_pt-20170623-114700-best",

    #                 "f92-0_pt-20170625-020639-best",
    #                 "f92-1_pt-20170625-021713-best",
    #                 "f92-2_pt-20170625-023812-best",
    #                 "f92-3_pt-20170625-020307-best",
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

]

#######################
# LOAD DATA
#######################
x_dict = build_joint_feature_vector(config_names)

# just to be sure
x = np.empty((len(x_dict),17*len(config_names)))

ids = x_dict.keys()
for i in range(len(ids)):
    x[i] = x_dict[ids[i]]

#######################
# LOAD MODEL
#######################

filename = "xgboost-allfold-"+"-".join(config_names)+".pkl"
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

file = open(os.path.join(metadata_dir,filename),"rb")
models, threshold = cPickle.load(file)
file.close()

#######################
# PREDICT
#######################

def predict_class(x,model):
    return model.predict(xgb.DMatrix(x))

test_preds = np.empty((x.shape[0],len(models),17))
for model_fold in range(len(models)):
    for f_idx in range(0, 17):
        print 'f_idx', f_idx
        test_preds[:, model_fold, f_idx] = predict_class(x, models[model_fold][f_idx])



imgid2raw = {}
for i in range(len(ids)):

    averaged = np.mean(test_preds[i],axis=0) > threshold
    imgid2raw[ids[i]] = averaged

#######################
# STORE
#######################
#
expid = "xgboost-allfolds-averaged-"+"-".join(config_names)
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
utils.auto_make_dir(outputs_path)
output_pickle_file = outputs_path + '/%s-%s.pkl' % (expid, "test-tta-xgboost")
file = open(output_pickle_file, "wb")
cPickle.dump(imgid2raw, file)
file.close()
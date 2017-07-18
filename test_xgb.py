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

def build_joint_prob_vector(config_names):
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

def build_joint_feature_vector(config_names,ids,pretext):

    vectors = {}


    for valid_id in ids:
        augmentations = []
        for aug in range(8):
            predictions = []
            for config_name in config_names:
                file = open(os.path.join("/data/plnt/model-predictions/fgodin/",
                                         config_name,"features",
                                         pretext+str(valid_id)+"_"+str(aug)+".npy"),"rb")
                predictions.append(np.load(file))
                file.close()

            augmentations.append(np.concatenate(predictions))

        vectors[valid_id] = augmentations


    return vectors


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





def get_all_pred(test_ids, pretext):
    #######################
    # LOAD DATA
    #######################
    x_dict = build_joint_feature_vector(config_names, test_ids, pretext)

    # just to be sure
    x = np.empty((len(x_dict),17*len(config_names)))

    i = 0
    augmentations = [np.empty((len(x_dict),7808)) for _ in range(8)]

    for key, vectors in x_dict.items():

        for j,vector in enumerate(vectors):
            augmentations[j][i] = vector

        i+=1
    #######################
    # PREDICT
    #######################

    def predict_class(x,f_idx):
        filename = "xgboost-features-" + str(f_idx) + "-".join(config_names) + ".pkl"
        metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

        bst = xgb.Booster({'nthread': 10})  # init model
        bst.load_model(os.path.join(metadata_dir, filename))  # load data

        preds = []
        for aug in range(len(x)):
            input = xgb.DMatrix(x[aug])
            preds.append(bst.predict(input))

        test = np.transpose(np.vstack(preds))
        return test

    test_preds = np.empty((augmentations[0].shape[0], 8, 17))

    for f_idx in range(0, 17):
        print 'f_idx', f_idx
        test_preds[:, :, f_idx] = predict_class(augmentations, f_idx)


    threshold = 0.24
    imgid2pred = {}
    imgid2raw = {}
    for i in range(len(test_ids)):

        averaged = np.mean(test_preds[i],axis=0) > threshold
        imgid2pred[pretext+str(test_ids[i])] = averaged
        imgid2raw[pretext + str(test_ids[i])] = test_preds[i]

    return imgid2pred, imgid2raw

test_ids = np.arange(40669)
test2_ids = np.arange(20522)

# test_ids = np.arange(10)
# test2_ids = np.arange(20)

test1_preds, test1_raw = get_all_pred(test_ids,"test_")
test2_preds, test2_raw = get_all_pred(test2_ids,"file")


#######################
# STORE
#######################





# correct my naming mistake
test2_rewrite = {}

for key,value in test2_raw.items():
    new_key = "file_"+key[4:]
    test2_rewrite[new_key]=value

imgid2raw = test1_raw.copy()
imgid2raw.update(test2_rewrite)



#
expid = "xgboost-features-fold4-averaged-"+"-".join(config_names)
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
utils.auto_make_dir(outputs_path)
output_pickle_file = outputs_path + '/%s-%s.pkl' % (expid, "test-tta-xgboost")
file = open(output_pickle_file, "wb")
cPickle.dump(imgid2raw, file)
file.close()
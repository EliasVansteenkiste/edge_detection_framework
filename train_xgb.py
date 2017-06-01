#! /usr/bin/python
import numpy as np
import xgboost as xgb
import os	
from sklearn import preprocessing

import utils
import app


seed = 31145

def load_features(path):
	img_ids = []
	filelist = os.listdir(path)
	for f in filelist:
		img_id = int(f.split('.')[0])
		img_ids.append(img_id)
	max_id = max(img_ids)

	test_features = utils.load_np(path + '/' + filelist[0])['features']
	features = np.zeros((max_id+1,)+test_features.shape)
	for f in filelist:
		img_id = int(f.split('.')[0])
		file = utils.load_np(features_path + '/' + f)
		features[img_id] = file['features']
	return features

def learn_weather_class(train_ids, valid_ids, features, labels):
	train_X = features[train_ids]
	valid_X = features[valid_ids]

	weather_labels = labels[:,:4] 
	print weather_labels[:10]
	weather_class = np.argmax(weather_labels, axis = 1)
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
	param['gamma']=0
	param['subsample']=0.8
	param['colsample_bytree']=0.8
	param['scale_pos_weight']=1
	#param['silent'] = 1
	param['nthread'] = 10
	param['num_class'] = 4

	watchlist = [ (xg_train,'train'), (xg_valid, 'valid') ]
	num_round = 60
	# bst = xgb.train(param, xg_train, num_round, watchlist );
	# # get prediction
	# pred = bst.predict( xg_valid );
	# print ('predicting, classification error=%f' % (sum( int(pred[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y)) ))

	# do the same thing again, but output probabilities
	param['objective'] = 'multi:softprob'
	bst = xgb.train(param, xg_train, num_round, watchlist );
	# Note: this convention has been changed since xgboost-unity
	# get prediction, this is in 1D array, need reshape to (ndata, nclass)
	yprob = bst.predict( xg_valid ).reshape( valid_Y.shape[0], 4)
	ylabel = np.argmax(yprob, axis=1)

	print 'f2_score', app.f2_score_arr(y_valid_true, yprob)
	y_argm = np.argmax(yprob,axis=1)
	y_max_pred = np.zeros_like(yprob)
	y_max_pred[np.arange(y_argm.shape[0]),y_argm] = 1
	print 'f2_score', app.f2_score_arr(y_valid_true, y_max_pred)
	print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y)) ))

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
	#param['eta'] = 0.1
	param['max_depth'] = 5
	param['n_estimators'] = 100
	param['learning_rate'] = 0.1
	param['min_child_weight'] = 1
	param['alpha'] = 0.01 # L1 regularization term on weights, default 0
	param['lambda'] = 0.01 # L2 regularization term on weights
	param['lambda_bias'] = 0.01 # L2 regularization term on bias, default 0
	param['gamma']=0
	#param['subsample']=0.8
	param['colsample_bytree']=0.95
	param['scale_pos_weight']=1
	#param['silent'] = 1
	param['nthread'] = 10
	param['num_class'] = 1

	watchlist = [ (xg_train,'train'), (xg_valid, 'valid') ]
	num_round = 30

	# do the same thing again, but output probabilities
	bst = xgb.train(param, xg_train, num_round, watchlist );
	# Note: this convention has been changed since xgboost-unity
	# get prediction, this is in 1D array, need reshape to (ndata, nclass)
	yprob = bst.predict( xg_valid )
	print yprob.shape
	ylabel = yprob > 0.5

	print 'skewed bce', app.logloss(yprob, valid_Y, skewing_factor = 5.)

	print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y)) ))


features_path = '/data/metadata/plnt/model-predictions/eavsteen/a14-20170510-070126'

features = load_features(features_path)

folds = app.make_stratified_split(no_folds=5)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]

labels = app.get_labels_array()
# learn_weather_class(train_ids, valid_ids, features, labels)
for f_idx in range(4,17):
	print 'f_idx', f_idx
	learn_bin_class(train_ids, valid_ids, features, f_idx, labels)

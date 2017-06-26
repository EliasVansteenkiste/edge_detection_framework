import utils
import pathfinder
import submission
import sys
import cPickle

import numpy as np

thresholds = eval("[ 0.3854,0.3222,0.5304,0.2979,0.5248,0.0609,0.6139,0.3466,0.675,0.6443,0.8752,0.553,0.5942,0.4224,0.5433,0.5197,0.5686]")
thresholds = np.asarray(thresholds)

config_names = ["f59_pt-20170617-000845"]
METADATA_PATH = "/data/plnt/"

metadata_dir = utils.get_dir_path('models', METADATA_PATH)

preds = []
for config_name in config_names:
    predictions_dir = utils.get_dir_path('model-predictions', METADATA_PATH)
    outputs_path = predictions_dir + '/' + config_name
    output_pickle_file = outputs_path + '/%s-%s.pkl' % (config_name, "test")
    predictions = cPickle.load(open(output_pickle_file,"rb"))
    preds.append(predictions)


imgid2pred = {}
keys = preds[0].keys()

for key in keys:
    for pred in preds:
        if key not in imgid2pred:
            imgid2pred[key] = [pred[key]]
        else:
            imgid2pred[key].append(pred[key])



for key,value in imgid2pred.items():

    values = np.amax(np.vstack(value),axis=0)


    imgid2pred[key] = values > thresholds



print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("f59_pt-average-thresholds", "test")
submission.write(imgid2pred, output_csv_file)
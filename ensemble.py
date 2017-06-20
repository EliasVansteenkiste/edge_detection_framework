import utils
import pathfinder
import submission
import sys
import cPickle
import numpy as np

config_names = ["f59_pt-20170617-000845","f59-0_pt-20170618-233610","f59-1_pt-20170619-075233","f59-2_pt-20170618-230254","f59-3_pt-20170619-065517"]


metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

preds = []
for config_name in config_names:
    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
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


    imgid2pred[key] = values > 0.5



print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("f59_pt-ensemble-max", "test")
submission.write(imgid2pred, output_csv_file)
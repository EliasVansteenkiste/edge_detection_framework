import utils
import pathfinder
import submission
import sys
import cPickle
import numpy as np

config_names = [#"f87_pt-20170623-114248-best",
                #"f87-0_pt-20170625-085603-best",
                #"f87-1_pt-20170625-085758-best",
                #"f87-2_pt-20170625-091432-best",
                #"f87-3_pt-20170625-083440-best",

                "f92_pt-20170623-114700-best",
                "f92-0_pt-20170625-020639-best",
                "f92-1_pt-20170625-021713-best",
                "f92-2_pt-20170625-023812-best",
                "f92-3_pt-20170625-020307-best",

                # "f97_pt-20170624-002331-best",
                # "f97-0_pt-20170624-180609-best",
                # "f97-1_pt-20170624-180713-best",
                # "f97-2_pt-20170624-180825-best",
                # "f97-3_pt-20170624-180932-best"
                ]

thresholds1 = np.asarray(eval("[ 0.4733,0.4841,0.5408,0.368,0.6024,0.2494,0.4868," \
             "0.4976,0.7461,0.534,0.6174,0.5359,0.5494,0.4419,0.5767,0.3928 ,0.5035]"))

thresholds2 = np.asarray(eval("[ 0.3834,0.3439,0.45,0.3892,0.5756,0.1702,0.419,0.5491,0.7933," \
              "0.5802,0.4961,0.468,0.5885,0.5853,0.4974,0.4825,0.6212]"))

thresholds3 = np.asarray(eval("[ 0.3789,0.4582,0.5741,0.4376,0.4046,0.1336,0.5841,0.6602,0.7492,"
                              "0.5713,0.249,0.5176,0.5403,0.5066,0.515,0.4683,0.5736]"))

thresholds = (thresholds1 + thresholds2 + thresholds3)/3

print(thresholds)

metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

preds = []
for config_name in config_names:
    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    outputs_path = predictions_dir + '/' + config_name
    output_pickle_file = outputs_path + '/%s-%s.pkl' % (config_name, "test_tta")
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

    values = np.mean(np.vstack(value),axis=0)

    imgid2pred[key] = values > 0.5


print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("f92-all-folds-ensemble", "test-tta")
submission.write(imgid2pred, output_csv_file)
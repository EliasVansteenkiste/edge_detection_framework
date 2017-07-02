import utils
import pathfinder
import submission
import sys
import cPickle
import numpy as np
from sklearn.metrics import fbeta_score

config_names = ["f87_pt-20170623-114248-best",
                "f87-0_pt-20170625-085603-best",
                "f87-1_pt-20170625-085758-best",
                "f87-2_pt-20170625-091432-best",
                "f87-3_pt-20170625-083440-best",

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

    return best_thresholds


metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

preds = []
thresholds = []
for config_name in config_names:
    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    outputs_path = predictions_dir + '/' + config_name
    output_pickle_file = outputs_path + '/%s-%s.pkl' % (config_name, "test_tta")
    predictions = cPickle.load(open(output_pickle_file,"rb"))
    preds.append(predictions)

    path_valid = "/data/plnt/model-predictions/fgodin/"+config_name+"/"+config_name+"_1_predictions.p"

    # file = open(path_valid, mode="rb")
    # preds_valid, targets_valid = cPickle.load(file)
    # file.close()
    #
    # thresholds.append(calculate_thresholds(preds_valid,targets_valid))

    thresholds.append([0.5]*17)


imgid2pred = {}
keys = preds[0].keys()

for key in keys:
    for pred in preds:
        if key not in imgid2pred:
            imgid2pred[key] = [pred[key]]
        else:
            imgid2pred[key].append(pred[key])


thresholds = np.vstack(thresholds)
print(thresholds)

i = 0
for key,value in imgid2pred.items():

    values = np.vstack(value) > thresholds
    result = np.zeros((values.shape[1],))

    if i < 5:
        print(key)
        print(values)
        i+=1

    for dim in range((values.shape[1])):
        result[dim] = np.bincount(values[:,dim]).argmax()

    imgid2pred[key] = result


print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("f87-f92-all-folds-ensemble-thresholds-majority-voting", "test-tta")
submission.write(imgid2pred, output_csv_file)
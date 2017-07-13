import utils
import pathfinder
import submission
import sys
import cPickle
import numpy as np
from sklearn.metrics import fbeta_score

config_names = [
"f87_pt-20170623-114248-best",
                "f87-0_pt-20170625-085603-best",
                "f87-1_pt-20170625-085758-best",
                "f87-2_pt-20170625-091432-best",
                "f87-3_pt-20170625-083440-best",

                "f92_pt-20170623-114700-best",
                "f92-0_pt-20170625-020639-best",
                "f92-1_pt-20170625-021713-best",
                "f92-2_pt-20170625-023812-best",
                "f92-3_pt-20170625-020307-best",

                "f95_pt-20170624-035637-best",
                "f95-0_pt-20170709-154037-best",
                "f95-1_pt-20170709-214044-best",
                "f95-2_pt-20170709-131550-best",
                "f95-3_pt-20170709-184153-best",

                "f97_pt-20170624-002331-best",
                "f97-0_pt-20170624-180609-best",
                "f97-1_pt-20170624-180713-best",
                "f97-2_pt-20170624-180825-best",
                "f97-3_pt-20170624-180932-best",

                    "f101_pt-20170626-090424-best",
                    "f101-0_pt-20170705-102142-best",
                    "f101-1_pt-20170705-165441-best",
                    "f101-2_pt-20170705-102109-best",
                    "f101-3_pt-20170705-165037-best"

                # "f113_pt-20170704-183933-best",
                # "f113-0_pt-20170703-231951-best",
                # "f113-1_pt-20170704-090405-best",
                # "f113-2_pt-20170703-232116-best",
                # "f113-3_pt-20170704-090759-best"


                ]

# config_names = [
#     "xgboost-allfolds-averaged-f87_pt-20170623-114248-best-f92_pt-20170623-114700-best-f97_pt-20170624-002331-best",
#     "xgboost-allfolds-averaged-f87-3_pt-20170625-083440-best-f92-3_pt-20170625-020307-best-f97-3_pt-20170624-180932-best",
#     "xgboost-allfolds-averaged-f87-2_pt-20170625-091432-best-f92-2_pt-20170625-023812-best-f97-2_pt-20170624-180825-best",
#     "xgboost-allfolds-averaged-f87-1_pt-20170625-085758-best-f92-1_pt-20170625-021713-best-f97-1_pt-20170624-180713-best",
#     "xgboost-allfolds-averaged-f87-0_pt-20170625-085603-best-f92-0_pt-20170625-020639-best-f97-0_pt-20170624-180609-best",
#
# ]

# config_names = [
#     "f119_pt-20170708-160916"
# ]

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

    print("Best F2: "+str(fbeta_score(targets, preds > best_thresholds, beta=2, average='samples')))

    return best_thresholds


metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)

preds = []
thresholds = []
i = 0
for config_name in config_names:
    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    outputs_path = predictions_dir + '/' + config_name
    output_pickle_file = outputs_path + '/%s-%s.pkl' % (config_name, "test_tta")
    predictions = cPickle.load(open(output_pickle_file,"rb"))
    preds.append(predictions)

    if config_name[0:4]=="f101" or config_name[0:3]=="f95":
        path_valid = "/data/plnt/model-predictions/fgodin/" + config_name + "/" + config_name + "_valid_tta_predictions.p"
    else:
        path_valid = "/data/plnt/model-predictions/fgodin/" + config_name + "/" + config_name + "_1_predictions.p"

    file = open(path_valid, mode="rb")
    preds_valid,targets_valid,ids_valid = cPickle.load(file)
    file.close()


    thresholds.append(calculate_thresholds(preds_valid,targets_valid))

    #thresholds.append([0.5]*17)


imgid2pred = {}
keys = preds[0].keys()

for key in keys:
    for pred in preds:
        if key not in imgid2pred:
            imgid2pred[key] = [pred[key]]
        else:
            imgid2pred[key].append(pred[key])


thresholds = np.vstack(thresholds)
#print(thresholds)

#thresholds = np.asarray([0.18,0.16,0.24,0.09,0.5,0.05,0.1,0.17,0.17,0.16,0.5,0.2,0.2,0.18,0.3,0.24,0.26])

i = 0
for key,value in imgid2pred.items():




    # break
    values = np.vstack(value) > thresholds
    #print(values)

    #result = np.asarray(values[0],dtype=np.int)

    #values = np.vstack(value)
    #print(values)
    # print(values.shape)



    result = np.zeros((values.shape[1],))
    # #
    # # if i < 5:
    # #     print(key)
    # #     print(values)
    # #     i+=1
    # #
    for dim in range((values.shape[1])):

        count = np.bincount(values[:,dim],minlength=2)
        result[dim] = 1 if count[1] > thresholds.shape[0]/2.0 else 0

    # print(result)
    # print(result.shape)
    # break

    imgid2pred[key] = result


print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("f87-f92-f95-f97-f101-adaptive-threshold-all-folds-ensembled-majority-voting", "test-tta")
submission.write(imgid2pred, output_csv_file)
import pathfinder
import utils
import cPickle
import numpy as np
import submission

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

expid = "xgboost-features-fold4-averaged-"+"-".join(config_names)
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
#utils.auto_make_dir(outputs_path)
output_pickle_file = outputs_path + '/%s-%s.pkl' % (expid, "test-tta-xgboost")
file = open(output_pickle_file, "rb")
imgid2raw = cPickle.load(file)
file.close()



def majority_vote(imgid2raw,threshold):
    imgid2pred = {}

    for key, value in imgid2raw.items():

        #print(value.shape)

        result = np.zeros((value.shape[1],))
        for dim in range(value.shape[1]):
            count = np.bincount(value[:,dim] > threshold,minlength=2)
            result[dim] = 1 if count[1] >= value.shape[0]/2.0 else 0
        imgid2pred[key] = result

    return imgid2pred

def average(imgid2raw,threshold):
    imgid2pred = {}

    for key, value in imgid2raw.items():

        imgid2pred[key] = np.mean(value,axis=0) > threshold

    return imgid2pred


imgid2pred = average(imgid2raw,0.24)



print 'writing submission'
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
output_csv_file = submissions_dir + '/%s-%s.csv' % ("xgboost-features-fold4-average-voting-"+"-".join(config_names), "test-tta")
submission.write(imgid2pred, output_csv_file)
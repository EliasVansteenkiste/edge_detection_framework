from sklearn.metrics import fbeta_score
import numpy as np
import cPickle

# based on https://drive.google.com/drive/folders/0B_DICebvRE-kWmFrbFUzbzhxMTQ


#path_train = "/data/plnt/model-predictions/fgodin/f59_pt-20170617-000845/f59_pt-20170617-000845_train_predictions.p"

path_valid1 = "/data/plnt/model-predictions/fgodin/f87_pt-20170623-114248-best/f87_pt-20170623-114248-best_1_predictions.p"

file = open(path_valid1,mode="rb")
preds1, targets1, ids1 = cPickle.load(file)
file.close()

path_valid2 = "/data/plnt/model-predictions/fgodin/f92_pt-20170623-114700-best/f92_pt-20170623-114700-best_1_predictions.p"

file = open(path_valid2,mode="rb")
preds2, targets2, ids2 = cPickle.load(file)
file.close()

path_valid3 = "/data/plnt/model-predictions/fgodin/f97_pt-20170624-002331-best/f97_pt-20170624-002331-best_1_predictions.p"

file = open(path_valid3,mode="rb")
preds3, targets3, ids3 = cPickle.load(file)
file.close()


pred_ensemble = np.zeros(preds1.shape)
preds2_new = np.empty(preds2.shape)
preds3_new = np.empty(preds3.shape)

for i in range(preds1.shape[0]):
    index1 = np.where(ids1[i]==ids2)[0][0]
    index2 = np.where(ids1[i] == ids3)[0][0]
    pred_ensemble[i] = (preds1[i,:]+preds2[index1,:]+preds3[index2,:])/3
    preds2_new[i]=preds2[index1,:]
    preds3_new[i] = preds3[index2, :]

preds2 = preds2_new
preds3 = preds3_new
targets2 = targets1
targets3 = targets1

indexes = np.arange(preds1.shape[0])
np.random.shuffle(indexes)


# path_valid3 = "/data/plnt/model-predictions/fgodin/f97_pt-20170624-002331-best/f97_pt-20170624-002331-best_1_predictions.p"
#
# file = open(path_valid3,mode="rb")
# preds3, targets3 = cPickle.load(file)
# file.close()

num_classes = 17

def calculate_perclass_f2(preds, targets, thresholds = [0.5] * num_classes):

    scores = []

    for i in range(preds.shape[1]):
        scores.append(fbeta_score(targets[:,i],preds[:,i]>thresholds[i],2))

    return scores

def calculate_f2(preds, targets, thresholds = [0.5] * num_classes):
    return fbeta_score(targets, preds > thresholds, beta=2, average='samples')

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

train_indexes = indexes[:indexes.shape[0]/2]
valid_indexes = indexes[indexes.shape[0]/2:]

print(calculate_perclass_f2(preds1[valid_indexes],targets1[valid_indexes]))
print(calculate_f2(preds1[valid_indexes],targets1[valid_indexes]))
print(calculate_perclass_f2(preds2[valid_indexes],targets2[valid_indexes]))
print(calculate_f2(preds2[valid_indexes],targets2[valid_indexes]))
print(calculate_perclass_f2(preds3[valid_indexes],targets3[valid_indexes]))
print(calculate_f2(preds3[valid_indexes],targets3[valid_indexes]))
print(calculate_perclass_f2(pred_ensemble[valid_indexes],targets1[valid_indexes]))
print(calculate_f2(pred_ensemble[valid_indexes],targets1[valid_indexes]))
print()

# thresholds=calculate_thresholds(preds1[train_indexes],targets1[train_indexes])
# print(calculate_perclass_f2(preds1[valid_indexes],targets1[valid_indexes],thresholds=thresholds))
# print(calculate_f2(preds1[valid_indexes],targets1[valid_indexes],thresholds=thresholds))
#
# thresholds=calculate_thresholds(preds2[train_indexes],targets2[train_indexes])
# print(calculate_perclass_f2(preds2[valid_indexes],targets2[valid_indexes],thresholds=thresholds))
# print(calculate_f2(preds2[valid_indexes],targets2[valid_indexes],thresholds=thresholds))
#
# thresholds=calculate_thresholds(preds3[train_indexes],targets3[train_indexes])
# print(calculate_perclass_f2(preds3[valid_indexes],targets3[valid_indexes],thresholds=thresholds))
# print(calculate_f2(preds3[valid_indexes],targets3[valid_indexes],thresholds=thresholds))
#
# thresholds=calculate_thresholds(pred_ensemble[train_indexes],targets1[train_indexes])
# print(calculate_perclass_f2(pred_ensemble[valid_indexes],targets1[valid_indexes],thresholds=thresholds))
# print(calculate_f2(pred_ensemble[valid_indexes],targets1[valid_indexes],thresholds=thresholds))


for i in  range(targets1.shape[0]):

    if targets1[i,16]==1:


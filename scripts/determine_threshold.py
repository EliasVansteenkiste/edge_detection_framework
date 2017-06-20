from sklearn.metrics import fbeta_score
import numpy as np
import cPickle

# based on https://drive.google.com/drive/folders/0B_DICebvRE-kWmFrbFUzbzhxMTQ


#path_train = "/data/plnt/model-predictions/fgodin/f59_pt-20170617-000845/f59_pt-20170617-000845_train_predictions.p"

path_valid = "/data/plnt/model-predictions/fgodin/f59_pt-20170617-000845/f59_pt-20170617-000845_valid_predictions.p"
file = open(path_valid,mode="rb")
preds, targets = cPickle.load(file)
file.close()

indexes = np.arange(preds.shape[0])
np.random.seed(123)
np.random.shuffle(indexes)

train_preds = preds[indexes[0:len(indexes)/2]]
train_targets = targets[indexes[0:len(indexes)/2]]

# train_preds = preds
# train_targets = targets

batch_size, num_classes = train_targets.shape[0:2]
num_iters = 100

best_thresholds = [0.2] * num_classes
best_scores = [0] * num_classes
for t in range(num_classes):

    thresholds = [0.2] * num_classes
    for i in range(num_iters):
        th = i / float(num_iters)
        thresholds[t] = th
        f2 = fbeta_score(train_targets, train_preds > thresholds, beta=2, average='samples')
        if f2 > best_scores[t]:
            best_scores[t] = f2
            best_thresholds[t] = th

print("Train F2 score: "+str(fbeta_score(train_targets, train_preds > best_thresholds, beta=2, average='samples')))

print(best_thresholds)


# file = open(path_valid,mode="rb")
# preds, targets = cPickle.load(file)
# file.close()
#
# valid_preds = preds
# valid_targets = targets

valid_preds = preds[indexes[len(indexes)/2:]]
valid_targets = targets[indexes[len(indexes)/2:]]

predictions = valid_preds > best_thresholds

print("Valid F2 score: "+str(fbeta_score(valid_targets, predictions, beta=2, average='samples')))


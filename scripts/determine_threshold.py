from sklearn.metrics import fbeta_score
import numpy as np
import cPickle

# based on https://drive.google.com/drive/folders/0B_DICebvRE-kWmFrbFUzbzhxMTQ


path_train = "/data/plnt/model-predictions/fgodin/f119_pt-20170708-160916/f119_pt-20170708-160916_train_tta_predictions.p"

file = open(path_train,mode="rb")
train_preds, train_targets, train_ids = cPickle.load(file)
file.close()

# path_valid = "/data/plnt/model-predictions/fgodin/f116_pt-20170707-231915-best/f116_pt-20170707-231915-best_valid_tta_predictions.p"
#
# file = open(path_valid,mode="rb")
# valid_preds, valid_targets, valid_ids = cPickle.load(file)
# file.close()



num_attempts = 100
np.random.seed(123456)

num_classes = 17


num_iters = 50

best_thresholds = [0.235] * num_classes
best_scores = [0] * num_classes


#np.random.shuffle(classes)

for t in range(num_classes):

    thresholds = [0.235] * num_classes
    for i in range(num_iters):
        ths = [0.5 - (i / float(num_iters)) * 0.5, 0.5 + (i / float(num_iters)) * 0.5]

        for th in ths:
            thresholds[t] = th
            f2 = fbeta_score(train_targets, train_preds > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
#
#
print("Train F2 score: "+str(fbeta_score(train_targets, train_preds > best_thresholds, beta=2, average='samples')))
#print("Valid F2 score: "+str(fbeta_score(valid_targets, valid_preds > best_thresholds, beta=2, average='samples')))

result="["
for i in range(len(best_thresholds)):
    result+=str(best_thresholds[i])+","

result=result[:-1]+"]"
print(result)
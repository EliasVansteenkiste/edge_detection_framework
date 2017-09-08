
import numpy as np

from collections import namedtuple
from functools import partial


from PIL import Image

import data_transforms
import data_iterators
import pathfinder
import utils
import app

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import tta

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform_train = {'vector_size': 6784,
               'channels': 0,
               'n_labels': 17,
               'num_aug': 8}

p_transform_valid = {'vector_size': 6784,
               'channels': 0,
               'n_labels': 17,
               'num_aug': 1}


#only lossless augmentations
p_augmentation = {
    'rot90_values': [0, 1, 2, 3],
    'flip': [0, 1]
}

# data preparation function
def data_prep_function_train(x, p_transform=p_transform_train, p_augmentation=p_augmentation, **kwargs):
    return x

def data_prep_function_valid(x, p_transform=p_transform_valid, **kwargs):
    return x

def label_prep_function(x):
    #cut out the label
    return x


# data iterators
batch_size = 32
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

folds = app.make_stratified_split(no_folds=10)
#for checking if folds are equal over multiple config files
for fold in folds:
    print sum(fold)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]+ folds[4] + folds[5] + folds[6]+ folds[7] + folds[8]
valid_ids = folds[9]
all_ids = []
for f in folds:
    all_ids = all_ids + f

bad_ids = []

train_ids = [x for x in train_ids if x not in bad_ids]
valid_ids = [x for x in valid_ids if x not in bad_ids]

test_ids = np.arange(40669)
test2_ids = np.arange(20522)

config_names = [
    "f87_f10-9_pt-20170717-185211-best",
    "f92-f10_9_pt-20170717-205233-best",
    "f95-f10_9_pt-20170717-012540-best",
    "f97_f10-9_pt-20170713-182024-best",
]



train_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform_train,
                                                    data_prep_fun = data_prep_function_train,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)

feat_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = all_ids,
                                                    p_transform=p_transform_train,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)

trainset_valid_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform_train,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)


valid_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)

test_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = test_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

test2_data_iterator = data_iterators.EnsembleDataGenerator(dataset=config_names,
                                                    batch_size=chunk_size,
                                                    img_ids = test2_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta = tta.LosslessTTA(p_augmentation)
tta_test_data_iterator = data_iterators.TTADataGenerator(dataset='test-jpg',
                                                    tta = tta,
                                                    duplicate_label = False,
                                                    img_ids = test_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta_test2_data_iterator = data_iterators.TTADataGenerator(dataset='test2-jpg',
                                                    tta = tta,
                                                    duplicate_label = False,
                                                    img_ids = test2_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta_valid_data_iterator = data_iterators.TTADataGenerator(dataset='train-jpg',
                                                    tta = tta,
                                                    duplicate_label = True,
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta_all_data_iterator = data_iterators.TTADataGenerator(dataset='train-jpg',
                                                    tta = tta,
                                                    duplicate_label = True,
                                                    batch_size=chunk_size,
                                                    img_ids = all_ids,
                                                    p_transform=p_transform_valid,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 40


validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(10 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-2,
    int(max_nchunks * 0.3): 2e-2,
    int(max_nchunks * 0.6): 1e-2,
    int(max_nchunks * 0.8): 3e-3,
    int(max_nchunks * 0.9): 1e-3
}

# model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier_drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(p_transform_train["vector_size"], p_transform_train["n_labels"])
        self.fc.weight.data.zero_()

    def forward(self, x):
        x = self.classifier_drop(x)

        return F.sigmoid(self.fc(x))


def build_model():
    net = Net()

    return namedtuple('Model', [ 'l_out'])( net )



# loss
class MultiLoss(torch.nn.modules.loss._Loss):

    def __init__(self, weight):
        super(MultiLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        torch.nn.modules.loss._assert_no_grad(target)
        weighted = (self.weight*target)*(input-target)**2 +(1-target)*(input-target)**2
        return torch.mean(weighted)


def build_objective():
    return MultiLoss(5.0)

def build_objective2():
    return MultiLoss(1.0)

def score(gts, preds):
    return app.f2_score_arr(gts, preds)

# updates
def build_updates(model, learning_rate):
    return optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.001)

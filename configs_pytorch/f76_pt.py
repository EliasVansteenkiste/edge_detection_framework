
#copy of j25
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

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (256, 256),
               'channels': 3,
               'n_labels': 17}


#only lossless augmentations
p_augmentation = {
    'rot90_values': [0,1,2,3],
    'flip': [0, 1]
}

# mean and std values for imagenet
mean=np.asarray([0.485, 0.456, 0.406])
mean = mean[:, None, None]
std = np.asarray([0.229, 0.224, 0.225])
std = std[:, None, None]

# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = x.convert('RGB')
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x -= mean
    x /= std
    x = x.astype(np.float32)
    x = data_transforms.lossless(x, p_augmentation, rng)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    x = x.convert('RGB')
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x -= mean
    x /= std
    x = x.astype(np.float32)
    return x

def label_prep_function(x):
    #cut out the label
    return x


# data iterators
batch_size = 32
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

folds = app.make_stratified_split(no_folds=5)
print len(folds)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]
all_ids = folds[0] + folds[1] + folds[2] + folds[3] + folds[4]

bad_ids = []

train_ids = [x for x in train_ids if x not in bad_ids]
valid_ids = [x for x in valid_ids if x not in bad_ids]

test_ids = np.arange(40669)
test2_ids = np.arange(20522)


train_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_train,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)

feat_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = all_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)

valid_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)

test_data_iterator = data_iterators.DataGenerator(dataset='test-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = test_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

test2_data_iterator = data_iterators.DataGenerator(dataset='test2-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = test2_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 60


validate_every = int(1 * nchunks_per_epoch)
save_every = int(10 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-1,
    int(max_nchunks * 0.3): 3e-2,
    int(max_nchunks * 0.6): 1e-2,
    int(max_nchunks * 0.8): 3e-3,
    int(max_nchunks * 0.9): 1e-3
}

# model
from collections import OrderedDict





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.densenet = torchvision.models.densenet169(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, p_transform["n_labels"])
        self.densenet.classifier.weight.data.zero_()

    def forward(self, x):
        x = self.densenet(x)
        return F.sigmoid(x)


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
        weighted_bce = - self.weight * target * torch.log(input + 1e-7) - (1 - target) * torch.log(1 - input + 1e-7)
        return torch.mean(weighted_bce)


def build_objective():
    return MultiLoss(5)

def build_objective2():
    return MultiLoss(1.0)

def score(gts, preds):
    return app.f2_score_arr(gts, preds)

# updates
def build_updates(model, learning_rate):
    return optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.0002)

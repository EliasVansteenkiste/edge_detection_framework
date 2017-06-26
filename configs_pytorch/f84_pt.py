
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
import torch.nn.init as init
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
    x = data_transforms.random_lossless(x, p_augmentation, rng)
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
max_nchunks = nchunks_per_epoch * 40


validate_every = int(1 * nchunks_per_epoch)
save_every = int(10 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-2,
    int(max_nchunks * 0.3): 2e-2,
    int(max_nchunks * 0.6): 1e-2,
    int(max_nchunks * 0.8): 3e-3,
    int(max_nchunks * 0.9): 1e-3
}

# model
from collections import OrderedDict
class MyDenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=p_transform["n_labels"]):

        super(MyDenseNet, self).__init__()
        self.num_classes = num_classes
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features

        final_num_features = 0
        for i, num_layers in enumerate(block_config):
            block = torchvision.models.densenet._DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = torchvision.models.densenet._Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        #self.classifier_drop = nn.Dropout(p=0.75)
        # Linear layer
        self.fc = nn.Linear(num_features,
                            num_classes * num_features / 8)
        init.orthogonal(self.fc.weight, gain=np.sqrt(2.0))

        self.classifiers = OrderedDict()


        for i in range(num_classes):
            c = nn.Linear(num_features / 8, 1)
            c.weight.data.zero_()
            c.cuda()
            self.classifiers[i] = c

        self.final_num_features = num_features / 8

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = self.classifier_drop(out)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.fc(out)
        out = F.relu(out, inplace=True)
        out_classes = []
        for i in range(self.num_classes):
            v = out.narrow(1, self.final_num_features*i, self.final_num_features)
            cl = self.classifiers[i]
            out_part = cl(v)
            out_part = F.sigmoid(out_part)
            out_classes.append(out_part)
        final_out = torch.cat(out_classes,dim=1)

        #temp = final_out.cpu().data.numpy()
        return final_out







def build_model():
    net = MyDenseNet()
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
    return MultiLoss(5.0)

def build_objective2():
    return MultiLoss(1.0)

def score(gts, preds):
    return app.f2_score_arr(gts, preds)

# updates
def build_updates(model, learning_rate):
    return optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.0005)

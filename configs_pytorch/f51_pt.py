
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
max_nchunks = nchunks_per_epoch * 40


validate_every = int(1 * nchunks_per_epoch)
save_every = int(5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-3,
    int(max_nchunks * 0.4): 5e-4,
    int(max_nchunks * 0.6): 2e-4,
    int(max_nchunks * 0.8): 1e-4,
    int(max_nchunks * 0.9): 5e-5
}

# model
class MyResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_drop = nn.Dropout(p=0.75)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc_drop(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def my_resnet34(pretrained=False, **kwargs):

    model = MyResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet34']))
    return model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = my_resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, p_transform["n_labels"])
        self.resnet.fc.weight.data.zero_()

    def forward(self, x):
        x = self.resnet(x)
        x_softmax = F.softmax(x.narrow(1,0,4))
        x_sigmoid = F.sigmoid(x.narrow(1,4,p_transform['n_labels']-4))
        x = torch.cat([x_softmax,x_sigmoid],1)

        return x


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
        softmax_weighted = self.weight * torch.sum(-torch.log(input.narrow(1,0,4)+1e-7)*target.narrow(1,0,4))

        bce_weighted = torch.sum(- self.weight * target.narrow(1,4,p_transform['n_labels']-4) *
                                 torch.log(input.narrow(1,4,p_transform['n_labels']-4)+1e-7)
                                 - (1 - target.narrow(1,4,p_transform['n_labels']-4)) *
                                 torch.log(1 - input.narrow(1,4,p_transform['n_labels']-4) + 1e-7))
        return (softmax_weighted + bce_weighted) / (p_transform["n_labels"]*batch_size)


def build_objective():
    return MultiLoss(5.0)

def build_objective2():
    return MultiLoss(1.0)

def score(gts, preds):
    return app.f2_score_arr(gts, preds)

# updates
def build_updates(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

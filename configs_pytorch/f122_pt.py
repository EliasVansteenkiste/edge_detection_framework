
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
import cPickle

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

import tta
tta = tta.LosslessTTA(p_augmentation)
tta_test_data_iterator = data_iterators.TTADataGenerator(dataset='test-jpg',
                                                    tta = tta,
                                                    duplicate_label = False,
                                                    img_ids = test_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta_test2_data_iterator = data_iterators.TTADataGenerator(dataset='test2-jpg',
                                                    tta = tta,
                                                    duplicate_label = False,
                                                    img_ids = test2_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

tta_valid_data_iterator = data_iterators.TTADataGenerator(dataset='train-jpg',
                                                    tta = tta,
                                                    duplicate_label = True,
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 10


validate_every = int(0.01 * nchunks_per_epoch)
save_every = int(10 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-3,
    int(max_nchunks * 0.4): 5e-4,
    int(max_nchunks * 0.7): 2e-4,
}

# model
from collections import OrderedDict
class MyDenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(MyDenseNet, self).__init__()



        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock


        num_features = num_init_features
        self.blocks = []
        self.transition = []
        final_num_features = 0




        self.denseblock1 = torchvision.models.densenet._DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        num_features = num_features + block_config[0] * growth_rate

        self.transition1 = torchvision.models.densenet._Transition(num_input_features=num_features,
                                                             num_output_features=num_features // 2)

        num_features = num_features // 2

        final_num_features+=num_features*16

        self.denseblock2 = torchvision.models.densenet._DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        num_features = num_features + block_config[1] * growth_rate

        self.transition2 = torchvision.models.densenet._Transition(num_input_features=num_features,
                                                             num_output_features=num_features // 2)

        num_features = num_features // 2

        final_num_features += num_features * 4

        self.denseblock3 = torchvision.models.densenet._DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        num_features = num_features + block_config[2] * growth_rate

        self.transition3 = torchvision.models.densenet._Transition(num_input_features=num_features,
                                                             num_output_features=num_features // 2)

        num_features = num_features // 2

        final_num_features += num_features

        self.denseblock4 = torchvision.models.densenet._DenseBlock(num_layers=block_config[3], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        num_features = num_features + block_config[3] * growth_rate

        final_num_features += num_features

        self.norm5 = nn.BatchNorm2d(num_features)

        #self.classifier_drop = nn.Dropout(p=0.75)

        self.classifier = nn.Linear(final_num_features, num_classes)

    def forward(self, x):

        x = self.features(x)
        b1 = self.denseblock1(x)
        t1 = self.transition1(b1)
        b2 = self.denseblock2(t1)
        t2 = self.transition2(b2)
        b3 = self.denseblock3(t2)
        t3 = self.transition3(b3)
        b4 = self.denseblock4(t3)
        t4 = F.relu(self.norm5(b4), inplace=True)

        t1_pooled = F.avg_pool2d(t1, kernel_size=7).view(batch_size, -1)
        t2_pooled = F.avg_pool2d(t2, kernel_size=7).view(batch_size, -1)
        t3_pooled = F.avg_pool2d(t3, kernel_size=7).view(batch_size, -1)
        t4_pooled = F.avg_pool2d(t4, kernel_size=7).view(batch_size, -1)

        catted = torch.cat([t1_pooled,t2_pooled,t3_pooled,t4_pooled],dim=1)

        temp = catted.cpu().data.numpy()

        #out = self.classifier_drop(catted)

        out = self.classifier(catted)
        return out


def my_densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MyDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32))

    return model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.densenet = my_densenet169()
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, p_transform["n_labels"])
        self.densenet.classifier.weight.data.zero_()

    def forward(self, x):
        x = self.densenet(x)
        return F.sigmoid(x)


def build_model():
    net = Net()

    filename = "/data/plnt/models/fgodin/f92_pt-20170623-114700-best.pkl"
    file = open(filename,"rb")
    model_params = cPickle.load(file)
    file.close()

    pretrained_dict = model_params['param_values']
    del pretrained_dict["densenet.classifier.weight"]

    for key in pretrained_dict.keys():
        if ".features.denseblock" in key:
            new_key = key.replace(".features.denseblock",".denseblock")
            pretrained_dict[new_key]=pretrained_dict[key]
            del pretrained_dict[key]
        elif ".features.transition" in key:
            new_key = key.replace(".features.transition", ".transition")
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]
        elif ".features.norm5" in key:
            new_key = key.replace(".features.norm5", ".norm5")
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]
    current_params = net.state_dict()
    current_params.update(pretrained_dict)

    net.load_state_dict(current_params)


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
    return optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.0002)

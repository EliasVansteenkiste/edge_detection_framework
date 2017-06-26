
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

torch.backends.cudnn.enabled=False

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
batch_size = 16
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

train_paths = app.get_image_paths(train_ids = train_ids,
                                  test_ids = test_ids, 
                                  test2_ids = test2_ids)

valid_paths = app.get_image_paths(train_ids = valid_ids)

test_paths = app.get_image_paths(test_ids = test_ids)
test2_paths = app.get_image_paths(test2_ids = test2_ids)

train_data_iterator = data_iterators.AutoEncoderDataGenerator(
                                                    batch_size=chunk_size,
                                                    img_paths = train_paths,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_train,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)


trainset_valid_data_iterator = data_iterators.AutoEncoderDataGenerator(
                                                    batch_size=chunk_size,
                                                    img_paths = valid_paths,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)


valid_data_iterator = data_iterators.AutoEncoderDataGenerator(
                                                    batch_size=chunk_size,
                                                    img_paths = valid_paths,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=True, infinite=False)

test_data_iterator = data_iterators.AutoEncoderDataGenerator(
                                                    batch_size=chunk_size,
                                                    img_paths = test_paths,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

test2_data_iterator = data_iterators.AutoEncoderDataGenerator(
                                                    batch_size=chunk_size,
                                                    img_paths = test2_paths,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 60


validate_every = int(0.1 * nchunks_per_epoch)
save_every = int(5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-3,
    int(max_nchunks * 0.4): 5e-4,
    int(max_nchunks * 0.6): 2e-4,
    int(max_nchunks * 0.8): 1e-4,
    int(max_nchunks * 0.9): 5e-5
}

# model definitions

class EncoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x) 

        out += residual
        out = self.relu(out)

        return out

class DecoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, outplanes, stride=1, upsample=None):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            output_padding=0
        elif stride == 2:
            output_padding=1
        else:
            raise NotImplementedError

        self.deconv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, output_padding=output_padding, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes
        self.outplanes = outplanes


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResAE(nn.Module):

    def __init__(self, encoder_block, decoder_block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices = True)
        

        self.layer1 = self._make_encoder_layer(encoder_block, 64, layers[0])
        self.fc_drop1 = nn.Dropout(p=0.5)
        self.layer2 = self._make_encoder_layer(encoder_block, 128, layers[1], stride=2)
        self.fc_drop2 = nn.Dropout(p=0.5)
        self.layer3 = self._make_encoder_layer(encoder_block, 256, layers[2], stride=2)
        self.fc_drop3 = nn.Dropout(p=0.5)
        self.layer4 = self._make_encoder_layer(encoder_block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc_drop4 = nn.Dropout(p=0.5)
        self.fc = nn.Linear( 2048*1*1, p_transform["n_labels"])
        

        self.fc_drop5 = nn.Dropout(p=0.5)
        self.layer5 = self._make_decoder_layer(decoder_block, 512, layers[3], outplanes = 1024, stride=2)
        self.fc_drop6 = nn.Dropout(p=0.5)
        self.layer6 = self._make_decoder_layer(decoder_block, 256, layers[2], outplanes = 512, stride=2)
        self.fc_drop7 = nn.Dropout(p=0.5)
        self.layer7 = self._make_decoder_layer(decoder_block, 128, layers[1], outplanes = 256, stride=2)
        self.fc_drop8 = nn.Dropout(p=0.5)
        self.layer8 = self._make_decoder_layer(decoder_block, 64, layers[0], outplanes = 64, stride=1)

        self.max_unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 7, stride=2, padding=3, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.c1_conv = nn.Conv2d(64, p_transform['channels'], kernel_size=1, stride=1, padding=0,
                               bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print 'downsample'
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

    def _make_decoder_layer(self, block, planes, blocks, outplanes, stride=1):
        layers = []
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.inplanes))
        
        upsample = None
        if stride != 1 or outplanes != self.inplanes:
            print 'upsample'
            if stride == 1:
                output_padding=0
            elif stride == 2:
                output_padding=1
            else:
                raise NotImplementedError

            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, outplanes,
                          kernel_size=1, stride=stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(outplanes),
            )
         
        layers.append(block(self.inplanes, planes, outplanes, stride, upsample))
        self.inplanes = outplanes

        return nn.Sequential(*layers)

    
    def forward(self, x, debug=False):
        # Encoder stage
        ## initial trunk
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        max_size = x.size()
        x, id1 = self.maxpool(x)

        ## residual blocks for the encoder
        x = self.layer1(x)
        if debug: print 'after layer1', x.size()
        x = self.fc_drop1(x)
        x = self.layer2(x)
        if debug: print 'after layer2', x.size()
        x = self.fc_drop2(x)
        x = self.layer3(x)
        if debug: print 'after layer3', x.size()
        x = self.fc_drop3(x)
        feats = self.layer4(x)



        # Classification output
        x = self.avgpool(feats)
        if debug: print 'after avgpool', x.size()
        x = x.view(x.size(0), -1)
        if debug: print x.size()
        x = self.fc_drop4(x)
        if debug: print x.size()
        x = self.fc(x)
        bc = F.sigmoid(x)

        # Decoder stage
        ## residual blocks for the decoder
        x = self.fc_drop5(feats)
        x = self.layer5(x)
        if debug: print 'after layer5', x.size()
        x = self.fc_drop6(x)
        x = self.layer6(x)
        if debug: print 'after layer6', x.size()
        x = self.fc_drop7(x)
        x = self.layer7(x)
        if debug: print 'after layer7', x.size()
        x = self.fc_drop8(x)
        x = self.layer8(x)

        ## final branch, mirroring initial trunk
        x = self.max_unpool(x, id1,max_size)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        reconstruction = self.c1_conv(x)


        return bc, reconstruction, feats


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resaenet = ResAE(EncoderBottleneck, DecoderBottleneck, [3, 4, 6, 3])
        self.resaenet.fc.weight.data.zero_()

    def forward(self, x):
        bc, reconstruction, feats = self.resaenet(x)
        return bc, reconstruction, feats


def build_model():
    net = Net()
    return namedtuple('Model', [ 'l_out'])( net )



# loss
class WeightedMultiLoss(torch.nn.modules.loss._Loss):

    def __init__(self, bce_weight):
        super(WeightedMultiLoss, self).__init__()
        self.bce_weight = bce_weight
    
    def forward(self, pred, reconstruction, target, original, has_label):
        torch.nn.modules.loss._assert_no_grad(original)
        torch.nn.modules.loss._assert_no_grad(target)
        torch.nn.modules.loss._assert_no_grad(has_label)

        weighted_bce = - self.bce_weight * target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
        weighted_bce = has_label * torch.squeeze(torch.mean(weighted_bce, dim=1))
        weighted_bce = torch.sum(weighted_bce) / torch.sum(has_label) /p_transform['n_labels']

        return weighted_bce

class ReconstructionError(torch.nn.modules.loss._Loss):

    def __init__(self, power=2):
        super(ReconstructionError, self).__init__()
        self.power = power

    def forward(self, pred, reconstruction, target, original, has_label):
        torch.nn.modules.loss._assert_no_grad(original)
        torch.nn.modules.loss._assert_no_grad(target)
        torch.nn.modules.loss._assert_no_grad(has_label)

        reconstruction_loss = (original - reconstruction) ** self.power
        reconstruction_loss = torch.mean(reconstruction_loss)

        return reconstruction_loss

class CombinedLoss(torch.nn.modules.loss._Loss):

    def __init__(self, bce_weight, alpha=.8, power=2.):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.alpha = alpha
        self.power = power


    def forward(self, pred, reconstruction, target, original, has_label):
        torch.nn.modules.loss._assert_no_grad(target)
        torch.nn.modules.loss._assert_no_grad(original)
        torch.nn.modules.loss._assert_no_grad(has_label)

        weighted_bce = - self.bce_weight * target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
        weighted_bce = has_label * torch.squeeze(torch.mean(weighted_bce, dim=1))
        weighted_bce = torch.sum(weighted_bce) / torch.sum(has_label) /p_transform['n_labels']

        reconstruction_loss = (original - reconstruction) **2
        reconstruction_loss = torch.mean(reconstruction_loss)

        loss = self.alpha * weighted_bce  + (1-self.alpha) * reconstruction_loss
        return loss


def build_objective():
    return CombinedLoss(bce_weight=5, alpha=.95, power=2.)

def build_objective2():
    return WeightedMultiLoss(5.)

def build_objective3():
    return ReconstructionError(2.)

def score(gts, preds):
    return app.f2_score_arr(gts, preds)

# updates
def build_updates(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)


#config a6 is equivalent to a5, except the normalization
import numpy as np
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T

import data_transforms
import data_iterators
import pathfinder
import utils
import app
import nn_planet

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (256, 256),
               'channels': 4,
               'n_labels': 17}


p_augmentation = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-32, 32),
    'do_flip': True,
    'allow_stretch': False,
}



# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x = x.astype(np.float32)
    x = data_transforms.perturb(x, p_augmentation, p_transform['patch_size'], rng, n_channels=p_transform['channels'])
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x = x.astype(np.float32)
    return x

def label_prep_function(label):
    return label


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
                                                    full_batch=False, random=False, infinite=False)

valid_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

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


validate_every = int(0.1 * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-4,
    int(max_nchunks * 0.4): 2e-4,
    int(max_nchunks * 0.6): 1e-4,
    int(max_nchunks * 0.7): 5e-5,
    int(max_nchunks * 0.8): 2e-5,
    int(max_nchunks * 0.9): 1e-5
}

# model
conv = partial(dnn.Conv2DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.HeNormal('relu'),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

conv1 = partial(dnn.Conv2DDNNLayer,
                 filter_size=1,
                 W=nn.init.Orthogonal(),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

up = partial(lasagne.layers.Upscale2DLayer,
                scale_factor=2)

concat = lasagne.layers.ConcatLayer


def inrn_v2(lin):
    n_base_filter = 32

    l1 = conv(lin, n_base_filter, filter_size=1)

    l2 = conv(lin, n_base_filter, filter_size=1)
    l2 = conv(l2, n_base_filter, filter_size=3)

    l3 = conv(lin, n_base_filter, filter_size=1)
    l3 = conv(l3, n_base_filter, filter_size=3)
    l3 = conv(l3, n_base_filter, filter_size=3)

    l = lasagne.layers.ConcatLayer([l1, l2, l3])

    l = conv(l, lin.output_shape[1], filter_size=1)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)

    return l


def inrn_v2_red(lin):
    # We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool(lin)

    l2 = conv(lin, ins // den * nom2, filter_size=3, stride=2)

    l3 = conv(lin, ins // den * nom2, filter_size=1)
    l3 = conv(l3, ins // den * nom3, filter_size=3, stride=2)

    l4 = conv(lin, ins // den * nom2, filter_size=1)
    l4 = conv(l4, ins // den * nom3, filter_size=3)
    l4 = conv(l4, ins // den * nom4, filter_size=3, stride=2)

    l = lasagne.layers.ConcatLayer([l1, l2, l3, l4])

    return l


def feat_red(lin):
    # We want to reduce the feature maps by a factor of 2
    ins = lin.output_shape[1]
    l = conv(lin, ins // 2, filter_size=1)
    return l

def LME(t_in, r=1.0, **kwargs):
    return T.log(T.mean(T.exp(r * t_in), **kwargs) + 1e-7) / r


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,p_transform['n_labels']))

    c1 = conv(l_in, 32)
    l = max_pool(c1)

    c2 = conv(l, 32)
    l = max_pool(c2)
    l = drop(l, 0.5)

    c3 = conv(l, 64)
    l = max_pool(c3)
    l = drop(l, 0.5)

    c4 = conv(l, 128)
    c5 = conv(c4, 128)
    c6 = conv(c5, 128)

    dc6 = drop(c6, 0.5)

    up1 = up(dc6)
    conc1 = concat([up1, c3])
    c7 = conv(conc1, 64)
    c8 = conv(c7, 64)

    up2 = up(c8)
    conc2 = concat([up2, c2])
    c9 = conv(conc2, 32)
    c10 = conv(c9, 32)

    up3 = up(c10)
    conc3 = concat([up3, c1])
    c11 = conv(conc3, 32)   

    l_seg_maps = conv1(c11, 13, nonlinearity=nn.nonlinearities.sigmoid)
    l_other = nn.layers.GlobalPoolLayer(l_seg_maps, pool_function=LME)

    l_feat = nn.layers.GlobalPoolLayer(l)
    l_weather = nn.layers.DenseLayer(l_feat, num_units=4,
                                 W=nn.init.Orthogonal(),
                                 b=nn.init.Constant(0.5),
                                 nonlinearity=nn.nonlinearities.softmax)
 

    l_out = nn.layers.ConcatLayer([l_weather, l_other], axis=-1)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target', 'l_feat', 'l_seg_maps'])(l_in, l_out, l_target, l_feat, l_seg_maps)


def build_objective(model, deterministic=False, epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = T.clip(predictions, epsilon, 1.-epsilon)
    #logs = [-T.log(preds), -T.log(1-preds)]
    weighted_bce = - 5 * targets * T.log(preds) - (1-targets)*T.log(1-preds)    
    reg = nn.regularization.l2(predictions)                                                                                                                                                                                       
    weight_decay=0.00004
    return T.mean(weighted_bce) + weight_decay * reg 

def build_objective2(model, deterministic=False, epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = T.clip(predictions, epsilon, 1.-epsilon)
    return T.mean(nn.objectives.binary_crossentropy(preds, targets))

def score(gts, preds, epsilon=1.e-11):
    return app.f2_score_arr(gts, preds)

test_score = score

def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates

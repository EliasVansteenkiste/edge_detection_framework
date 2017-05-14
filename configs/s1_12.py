
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

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (256, 256),
               'channels': 4,
               'n_labels': 1}


p_augmentation = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-32, 32),
    'do_flip': True,
    'allow_stretch': False,
}


channel_norm_stats = {
    0.1: [2739., 2022., 1284., 1091.],
    0.5: [3016., 2272., 1433., 1415.],
    1: [3149., 2441., 1563.,  1733.],
    5: [3514., 2867., 1792., 3172.],
    10: [3661., 3016., 1902., 4132.],
    50: [4503., 3768., 2534., 6399.],
    90: [6615., 5912., 4694., 8311.],
    95: [7623., 6822., 5698., 9109.],
    99: [11065., 10184., 9047., 11561.],
    99.5: [14686., 13508., 12197., 12820.],
    99.9: [23722., 16926., 19183., 16523.]}

# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = x.astype(np.float32)
    x = data_transforms.perturb(x, p_augmentation, p_transform['patch_size'], rng)
    x = data_transforms.channel_norm(x, img_stats = channel_norm_stats, percentiles=[.1,99.9], no_channels=4)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    #take a patch in the middle of the chip
    x = x.astype(np.float32)
    x = data_transforms.channel_norm(x, img_stats = channel_norm_stats, percentiles=[.1,99.9], no_channels=4)
    return x

label_id = 12
def label_prep_function(x):
    #cut out the label
    return x[label_id]

# data iterators
batch_size = 16
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

folds = app.make_stratified_split(no_folds=5)
print len(folds)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]

bad_ids = []

train_ids = [x for x in train_ids if x not in bad_ids]
valid_ids = [x for x in valid_ids if x not in bad_ids]




train_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_train,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)


valid_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 40


validate_every = int(.25 * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

print 'validate_every', validate_every
print 'save_every', save_every

learning_rate_schedule = {
    0: 2e-4,
    int(max_nchunks * 0.4): 1e-4,
    int(max_nchunks * 0.6): 5e-5,
    int(max_nchunks * 0.7): 2e-5,
    int(max_nchunks * 0.8): 1e-5,
    int(max_nchunks * 0.9): 5e-6
}

# model
conv = partial(dnn.Conv2DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


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


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,))

    l = conv(l_in, 64)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = drop(l)
    l = nn.layers.GlobalPoolLayer(l)


    l_out = nn.layers.DenseLayer(l, num_units=1,
                                 W=nn.init.Orthogonal(),
                                 b=nn.init.Constant(0.5),
                                 nonlinearity=nn.nonlinearities.sigmoid)


    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


biases = [ 0.69673164,  0.17912992,  0.06657773,  0.05756071,  0.00516317,  0.00247042,
  0.02122088,  0.00837471,  0.00820178,  0.00839942,  0.00242101, 0.30480002,
  0.11060056,  0.09046666,  0.9348057,   0.19951086,  0.17940167]

def build_objective(model, deterministic=False, epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = T.clip(predictions, epsilon, 1.-epsilon)
    #logs = [-T.log(preds), -T.log(1-preds)]
    weighted_bce = - 5 * targets * T.log(preds) - (1-targets)*T.log(1-preds)    
    reg = nn.regularization.l2(predictions)                                                                                                                                                                                       
    weight_decay=0.00004
    return T.mean(weighted_bce) + weight_decay * reg 


def build_objective2(model, deterministic=False, bias = app.get_biases()[label_id], epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = predictions*np.float32(0.5/bias)
    preds = T.clip(preds, epsilon, 1.-epsilon)
    #logs = [-T.log(preds), -T.log(1-preds)]
    weighted_bce = - 5 * targets * T.log(preds) - (1-targets)*T.log(1-preds)    
    reg = nn.regularization.l2(predictions)                                                                                                                                                                                       
    weight_decay=0.00004
    return T.mean(weighted_bce) + weight_decay * reg 

def build_test_objective(model, deterministic=False, bias = app.get_biases()[label_id],  epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = predictions*np.float32(0.5/bias)
    preds = T.clip(preds, epsilon, 1.-epsilon)
    return T.mean(nn.objectives.binary_crossentropy(preds, targets))

def score(gts, preds):
    treshold = 0.5
    preds_cutoff = np.array([1 if p>treshold else 0 for p in preds])
    gts  = np.array(gts)
    tp = np.sum(gts*preds_cutoff)
    fp = np.sum((1-gts)*preds_cutoff)
    fn = np.sum(gts*(1-preds_cutoff))
    return np.array([tp, fp, fn])

def test_score(gts, preds, bias = app.get_biases()[label_id], epsilon=1.e-11):
    predictions = np.array(preds)/0.5*bias
    preds_cutoff = [1 if p>0.5 else 0 for p in predictions]
    return app.f2_score(gts, preds_cutoff, average=None)

def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates

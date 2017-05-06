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
               'n_labels': 17}


p_augmentation = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-32, 32),
    'do_flip': True,
    'allow_stretch': False,
}


channel_zmuv_stats = {
    'avg': [4970.55, 4245.35, 3064.64, 6360.08],
    'std': [1785.79, 1576.31, 1661.19, 1841.09]}

# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = x.astype(np.float32)
    x = data_transforms.perturb(x, p_augmentation, p_transform['patch_size'], rng)
    x = data_transforms.channel_zmuv(x, img_stats = channel_zmuv_stats, no_channels=4)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    #take a patch in the middle of the chip
    x = x.astype(np.float32)
    x = data_transforms.channel_zmuv(x, img_stats = channel_zmuv_stats, no_channels=4)
    return x


# data iterators
batch_size = 8
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

folds = app.make_stratified_split(no_folds=5)
print len(folds)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]

bad_ids = [18772, 28173, 5023]

train_ids = [x for x in train_ids if x not in bad_ids]
valid_ids = [x for x in valid_ids if x not in bad_ids]


train_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_train,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100


validate_every = int(0.1 * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-4,
    int(max_nchunks * 0.5): 2e-4,
    int(max_nchunks * 0.6): 1e-4,
    int(max_nchunks * 0.7): 5e-5,
    int(max_nchunks * 0.8): 2e-5,
    int(max_nchunks * 0.9): 1e-5
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
batch_norm = lasagne.layers.batch_norm
ConvLayer = lasagne.layers.Conv2DLayer
rectify = lasagne.nonlinearities.rectify

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block(l, increase_dim=False, projection=False):
    input_num_filters = l.output_shape[1]
    if increase_dim:
        first_stride = (2,2)
        out_num_filters = input_num_filters*2
    else:
        first_stride = (1,1)
        out_num_filters = input_num_filters

    stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
            block = nn.layers.NonlinearityLayer(nn.layers.ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            identity = nn.layers.ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
            padding = nn.layers.PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
            block = nn.layers.NonlinearityLayer(nn.layers.ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
    else:
        block = nn.layers.NonlinearityLayer(nn.layers.ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
    
    return block



def build_model(l_in=None):


    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,p_transform['n_labels']))

    n=5

    # first layer, output is 64 x 256 x 256
    l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 128 x 128 x 128
    l = residual_block(l, increase_dim=True)

    # second stack of residual blocks, output is 256 x 64 x 64
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 512 x 32 x 32
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # fourth stack of residual blocks, output is 1024 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = nn.layers.GlobalPoolLayer(l)

    l_out = nn.layers.DenseLayer(l, num_units=p_transform['n_labels'],
                                 W=nn.init.Orthogonal(),
                                 b=nn.init.Constant(0.5),
                                 nonlinearity=nn.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    preds = T.clip(predictions, epsilon, 1.-epsilon)
    return T.mean(nn.objectives.binary_crossentropy(preds, targets))

def build_objective2(model, deterministic=False, epsilon=1.e-7):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    
    tp = targets * predictions
    fn = targets * (1.-predictions)
    fp = (1.-targets) * predictions

    f2 = 5.*tp/(5.*tp+4.*fn+fp+epsilon)

    return T.mean(f2)

def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates

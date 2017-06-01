
#config a6 is equivalent to a5, except the normalization
import numpy as np
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                            DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                            NonlinearityLayer, ScaleLayer, BiasLayer)
from lasagne.nonlinearities import rectify, softmax
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer



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

def label_prep_function_train(label):
    new_label = np.copy(label)
    if label[12]:
        new_label[11] = 1
    return new_label

def label_prep_function_valid(label):
    return label


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
                                                    label_prep_fun = label_prep_function_train,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)

feat_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = all_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function_valid,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

valid_data_iterator = data_iterators.DataGenerator(dataset='train-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function_valid,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

test_data_iterator = data_iterators.DataGenerator(dataset='test-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = test_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function_valid,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

test2_data_iterator = data_iterators.DataGenerator(dataset='test2-jpg',
                                                    batch_size=chunk_size,
                                                    img_ids = test2_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    label_prep_fun = label_prep_function_valid,
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
                 W=nn.init.Orthogonal(),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)




def affine_relu_conv(network, channels, filter_size, dropout, name_prefix):
    network = ScaleLayer(network, name=name_prefix + '_scale')
    network = BiasLayer(network, name=name_prefix + '_shift')
    network = NonlinearityLayer(network, nonlinearity=rectify,
                                name=name_prefix + '_relu')
    network = Conv2DLayer(network, channels, filter_size, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv')
    if dropout:
        network = DropoutLayer(network, dropout)
    return network

def transition(network, dropout, name_prefix):
    # a transition 1x1 convolution followed by avg-pooling
    network = affine_relu_conv(network, channels=network.output_shape[1],
                               filter_size=1, dropout=dropout,
                               name_prefix=name_prefix)
    network = Pool2DLayer(network, 2, mode='average_inc_pad',
                          name=name_prefix + '_pool')
    network = BatchNormLayer(network, name=name_prefix + '_bn',
                             beta=None, gamma=None)
    return network

def dense_block(network, num_layers, growth_rate, dropout, name_prefix):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = affine_relu_conv(network, channels=growth_rate,
                                filter_size=3, dropout=dropout,
                                name_prefix=name_prefix + '_l%02d' % (n + 1))
        conv = BatchNormLayer(conv, name=name_prefix + '_l%02dbn' % (n + 1),
                              beta=None, gamma=None)
        network = ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network

def build_densenet(l_in, input_var=None, first_output=64, growth_rate=32, num_blocks=4, dropout=0):
    """
    Creates a DenseNet model in Lasagne.
    Parameters
    ----------
    input_shape : tuple
        The shape of the input layer, as ``(batchsize, channels, rows, cols)``.
        Any entry except ``channels`` can be ``None`` to indicate free size.
    input_var : Theano expression or None
        Symbolic input variable. Will be created automatically if not given.
    classes : int
        The number of classes of the softmax output.
    first_output : int
        Number of channels of initial convolution before entering the first
        dense block, should be of comparable size to `growth_rate`.
    growth_rate : int
        Number of feature maps added per layer.
    num_blocks : int
        Number of dense blocks (defaults to 3, as in the original paper).
    dropout : float
        The dropout rate. Set to zero (the default) to disable dropout.
    batchsize : int or None
        The batch size to build the model for, or ``None`` (the default) to
        allow any batch size.
    inputsize : int, tuple of int or None
    Returns
    -------
    network : Layer instance
        Lasagne Layer instance for the output layer.
    References
    ----------
    .. [1] Gao Huang et al. (2016):
           Densely Connected Convolutional Networks.
           https://arxiv.org/abs/1608.06993
    """
    

    nb_layers = [6, 12, 32, 32] # For DenseNet-169
    nb_layers = [6, 12, 24, 16] # For DenseNet-121
    # initial convolution
    network = Conv2DLayer(l_in, first_output, filter_size=7, stride=2, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    network = BatchNormLayer(network, name='pre_bn', beta=None, gamma=None)
    network = ScaleLayer(network, name='pre_scale')
    network = BiasLayer(network, name='pre_shift')
    network = dnn.MaxPool2DDNNLayer(network, pool_size=3, stride=2) 
    # note: The authors' implementation does *not* have a dropout after the
    #       initial convolution. This was missing in the paper, but important.
    # if dropout:
    #     network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between

    for b in range(num_blocks):
        network = dense_block(network, nb_layers[b], growth_rate, dropout,
                              name_prefix='block%d' % (b + 1))
        if b < num_blocks - 1:
            network = transition(network, dropout,
                                 name_prefix='block%d_trs' % (b + 1))
    # post processing until prediction
    network = ScaleLayer(network, name='post_scale')
    network = BiasLayer(network, name='post_shift')
    network = NonlinearityLayer(network, nonlinearity=rectify, name='post_relu')

    return network

def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,p_transform['n_labels']))

    l_feat_cube = build_densenet(l_in, input_var=None, 
                             first_output=64, 
                             growth_rate=32, 
                             num_blocks=4,
                             dropout=0)
    l_d = drop(l_feat_cube)
    l_feat = GlobalPoolLayer(l_d, name='post_pool')


    l = nn.layers.DenseLayer(l_feat, num_units=p_transform['n_labels'],
                                 W=nn.init.HeNormal(gain=1),
                                 nonlinearity=nn.nonlinearities.identity)


    l_weather = nn.layers.SliceLayer(l, indices=slice(0,4), axis=-1)
    l_weather = nn.layers.NonlinearityLayer(l_weather, nonlinearity=nn.nonlinearities.softmax)

    l_other = nn_planet.MajorExclusivityLayer(l, idx_major=0)
    l_other = nn.layers.SliceLayer(l_other, indices=slice(4,None), axis=-1)

    l_out = nn.layers.ConcatLayer([l_weather, l_other], axis=-1)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target', 'l_feat'])(l_in, l_out, l_target, l_feat)


def build_objective(model, deterministic=False, epsilon=1.e-7):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_target)
    preds = T.clip(predictions, epsilon, 1.-epsilon)

    p_w = preds[:,:4]
    t_w = targets[:,:4]
    p_o = preds[:,4:]
    t_o = targets[:,4:]
    weighted_ce = - 5 * t_w * T.log(p_w)
    weighted_bce = - 5 * t_o * T.log(p_o) - (1-t_o)*T.log(1-p_o)    
    cost_fu = T.concatenate([weighted_ce, weighted_bce],axis=1)
    reg = nn.regularization.l2(predictions)                                                                                                                                                                                       
    weight_decay=0.00004
    return T.mean(cost_fu) + weight_decay * reg 

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

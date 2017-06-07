
#config a6 is equivalent to a5, except the normalization
import numpy as np
from collections import namedtuple
from functools import partial
from PIL import Image
from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import optimizers
import keras

from keras_custom_layers import Scale
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
               'channels': 3,
               'n_labels': 17}


#only lossless augmentations
p_augmentation = {
    'rot90_values': [0,1,2,3],
    'flip': [0, 1]
}



# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = x.convert('RGB')
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x = x.astype(np.float32)
    x = data_transforms.lossless(x, p_augmentation, rng)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    x = x.convert('RGB')
    x = np.array(x)
    x = np.swapaxes(x,0,2)
    x = x / 255.
    x = x.astype(np.float32)
    return x

def label_prep_function(label):
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

def conv(x, n_filters, name, eps = np.float32(1.1e-5), concat_axis=1):
    x = ZeroPadding2D((1, 1), name='zp_conv_'+name)(x)
    x = Convolution2D(n_filters, 3, 3, subsample=(1, 1), name='conv_'+name, bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_bn_'+name)(x)
    x = Scale(axis=concat_axis, name='conv_scale_'+name)(x)
    x = Activation('relu', name='conv_relu_'+name)(x)
    return x


def build_model():
    
    eps = np.float32(1.1e-5)
    concat_axis = 1

    l_in = Input(shape=(p_transform['channels'],) + p_transform['patch_size'], name='data')

    x = ZeroPadding2D((3, 3), name='conv0_zeropadding')(l_in)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv0', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv0_bn')(x)
    x = Scale(axis=concat_axis, name='conv0_scale')(x)
    x = Activation('relu', name='relu0')(x)

    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv(x, 64, '1')
    x = conv(x, 64, '2')

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x = conv(x, 128, '3')
    x = conv(x, 128, '4')

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = conv(x, 128, '5')
    x = conv(x, 128, '6')

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    x = conv(x, 128, '7')
    x = conv(x, 128, '8')

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = Dropout(0.5)(x)

    l_feat = GlobalAveragePooling2D(name='global_avg_pool')(x)

    l_weather = Dense(4, name='dense_weather')(l_feat)
    l_weather = Activation('softmax', name='softmax_weather')(l_weather)

    l_other = Dense(13, name='dense_other')(l_feat)
    l_other = Activation('sigmoid', name='sigmoid_other')(l_other)

    l_out = merge([l_weather, l_other], mode='concat', concat_axis=1, name='concat_output_nodes')

    model = Model(input=l_in, output=l_out)
    return model

    #return namedtuple('Model', ['model', 'l_in', 'l_out', 'l_target', 'l_feat'])(model, l_in, l_out, l_target, l_feat)



def l2(x):
    """Computes the squared L2 norm of a tensor
    Parameters
    ----------
    x : tensor
    Returns
    -------
    scalar
        squared l2 norm (sum of squared values of elements)
    """
    return K.sum(x**2)


def loss(y_true, y_pred, epsilon=1.e-7):
    predictions = K.flatten(y_pred)
    targets = K.flatten(y_true)
    preds = K.clip(predictions, epsilon, 1.-epsilon)
    weighted_bce = - 5 * targets * K.log(preds) - (1-targets)*K.log(1-preds)    
    reg = l2(predictions)                                                                                                                                                                                       
    weight_decay=0.00004
    return K.mean(weighted_bce, axis=-1) + weight_decay * reg 


def loss2(y_true, y_pred, epsilon=1.e-7):
    predictions = K.flatten(y_pred)
    targets = K.flatten(y_true)
    preds = K.clip(predictions, epsilon, 1.-epsilon)
    return K.mean(K.binary_crossentropy(preds, targets), axis=-1)

def score(gts, preds, epsilon=1.e-11):
    return app.f2_score_arr(gts, preds)

test_score = score

def optimizer(learning_rate):
    adam = optimizers.Adam(lr=learning_rate)
    return adam

import numpy as np
import itertools

import pathfinder
import utils
import app
import buffering
import os

class DataGenerator(object):
    def __init__(self, dataset, batch_size, img_ids, p_transform, data_prep_fun, label_prep_fun, rng,
                 random, infinite, full_batch, override_patch_size=None, version=1, **kwargs):


        self.dataset = dataset
        self.img_ids = img_ids
        self.nsamples = len(img_ids)
        self.batch_size = batch_size
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
        self.label_prep_fun = label_prep_fun
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.full_batch = full_batch
        self.override_patch_size = override_patch_size
        if override_patch_size:
            self.patch_size = override_patch_size
        else:
            self.patch_size = self.p_transform['patch_size']


        self.labels = app.get_labels_array(version=version)

    def generate(self):
        while True:
            rand_idxs = np.arange(len(self.img_ids))
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                if self.p_transform['channels']:
                    x_batch = np.zeros((nb,self.p_transform['channels'],) + self.patch_size, dtype='float32')
                else:
                    x_batch = np.zeros((nb,) + self.patch_size, dtype='float32')
                if self.p_transform['n_labels']>1:
                    y_batch = np.zeros((nb, self.p_transform['n_labels']), dtype='float32')
                else:
                    y_batch = np.zeros((nb,), dtype='float32')

                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_id = self.img_ids[idx]
                    batch_ids.append(img_id)
                    try:
                        img = app.read_compressed_image(self.dataset, img_id)
                    except Exception:
                        print 'cannot open ', img_id
                    x_batch[i] = self.data_prep_fun(x=img)
                    if 'train' in self.dataset:
                        y_batch[i] = self.label_prep_fun(self.labels[img_id])

                    #print 'i', i, 'img_id', img_id, y_batch[i]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break

class AutoEncoderDataGenerator(object):
    def __init__(self, batch_size, img_paths, labeled_img_paths, p_transform, data_prep_fun, label_prep_fun, rng,
                 random, infinite, full_batch, **kwargs):


        self.img_paths = img_paths
        self.labeled_img_paths = labeled_img_paths
        self.nsamples = len(img_paths)
        self.batch_size = batch_size
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
        self.label_prep_fun = label_prep_fun
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.full_batch = full_batch

        self.labels = app.get_labels_array()

    def generate(self):
        while True:
            rand_idxs = np.arange(len(self.img_paths))
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                if self.p_transform['channels']:
                    x_batch = np.zeros((nb,self.p_transform['channels'],) + self.p_transform['patch_size'], dtype='float32')
                else:
                    x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
                if self.p_transform['n_labels']>1:
                    y_batch = np.zeros((nb, self.p_transform['n_labels']), dtype='float32')
                else:
                    y_batch = np.zeros((nb,), dtype='float32')
                z_batch = np.zeros((nb,), dtype='float32')

                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_path = self.img_paths[idx]
                    batch_ids.append(img_path)
                    try:
                        img = app.read_image_from_path(img_path)
                    except Exception:
                        print 'cannot open ', img_id
                    x_batch[i] = self.data_prep_fun(x=img)
                    
                    if img_path in self.labeled_img_paths:
                        z_batch[i] = 1.
                        img_id = app.get_id_from_path(img_path)
                        y_batch[i] = self.label_prep_fun(self.labels[img_id])

                    #print 'i', i, 'img_id', img_id, y_batch[i]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, z_batch, batch_ids
                else:
                    yield x_batch, y_batch, z_batch, batch_ids

            if not self.infinite:
                break

class SlimDataGenerator(object):
    def __init__(self, dataset, batch_size, img_ids, p_transform, data_prep_fun, label_prep_fun, rng,
                 random, infinite, full_batch, **kwargs):


        self.dataset = dataset
        self.img_ids = img_ids
        self.nsamples = len(img_ids)
        self.batch_size = batch_size
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
        self.label_prep_fun = label_prep_fun
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.full_batch = full_batch

        self.labels = app.get_labels_array()

    def generate(self):
        while True:
            rand_idxs = np.arange(len(self.img_ids))
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = []
                y_batch = []
                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_id = self.img_ids[idx]
                    batch_ids.append(img_id)
                    try:
                        img = app.read_compressed_image(self.dataset, img_id)
                    except Exception:
                        print 'cannot open ', img_id
                    x_batch.append(self.data_prep_fun(x=img))
                    if 'train' in self.dataset:
                        y_batch.append(self.label_prep_fun(self.labels[img_id]))

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break

def _test_data_generator():
        #testing data iterator 

    p_transform = {'patch_size': (256, 256),
               'channels': 4,
               'n_labels': 17}
    rng = np.random.RandomState(42)

    def data_prep_fun(x):
        x = np.array(x)
        x = np.swapaxes(x,0,2)
        return x

    def label_prep_fun(labels):
        return labels

    folds = app.make_stratified_split(no_folds=5)
    all_ids = folds[0] + folds[1] + folds[2] + folds[3] +folds[4]
    bad_ids = [18772, 28173, 5023]
    img_ids = [x for x in all_ids if x not in bad_ids]

    dg = DataGenerator(dataset='train-jpg',
                        batch_size=10,
                        img_ids = img_ids,
                        p_transform=p_transform,
                        data_prep_fun = data_prep_fun,
                        label_prep_fun = label_prep_fun,
                        rng=rng,
                        full_batch=True, random=False, infinite=False)

    for (x_chunk, y_chunk, id_train) in buffering.buffered_gen_threaded(dg.generate()):
        print x_chunk.shape, y_chunk.shape, id_train


def _test_simple_data_generator():
        #testing data iterator 

    p_transform = {'patch_size': (256, 256),
               'channels': 4,
               'n_labels': 1}

    label_id = 4
    rng = np.random.RandomState(42)

    def data_prep_fun(x):
        return x

    def label_prep_fun(labels):
        print labels
        return labels[label_id]

    folds = app.make_stratified_split(no_folds=5)
    all_ids = folds[0] + folds[1] + folds[2] + folds[3] +folds[4]
    bad_ids = []
    img_ids = [x for x in all_ids if x not in bad_ids]

    dg = SlimDataGenerator(dataset='train-jpg',
                                batch_size=10,
                                label_id = label_id,
                                img_ids = all_ids,
                                p_transform=p_transform,
                                data_prep_fun = data_prep_fun,
                                label_prep_fun = label_prep_fun,
                                rng=rng, 
                                full_batch=True, 
                                random=False, 
                                infinite=False)

    print 'start'
    avgs = []
    stds = []

    ch_avgs = [[],[],[],[]]
    ch_stds = [[],[],[],[]]
    for (x_chunk, y_chunk, id_train) in dg.generate():
        x_chunk = np.stack(x_chunk)
        #x_chunk = x_chunk/255.
        avgs.append(np.mean(x_chunk))
        stds.append(np.std(x_chunk))
        for ch in range(4):
            ch_avgs[ch].append(np.mean(x_chunk[:,ch]))
            ch_stds[ch].append(np.std(x_chunk[:,ch]))

    print 'avgs', np.mean(np.stack(avgs))
    print 'stds', np.mean(np.stack(stds))
    for ch in range(4):
        print 'ch', str(ch)
        print 'mean of avgs', np.mean(ch_avgs[ch])
        print 'mean of stds', np.mean(ch_stds[ch])
        

if __name__ == "__main__":
    _test_simple_data_generator()


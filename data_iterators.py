import numpy as np
import itertools

import pathfinder
import utils
import app
import buffering

class DataGenerator(object):
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
                if self.p_transform['channels']:
                    x_batch = np.zeros((nb,self.p_transform['channels'],) + self.p_transform['patch_size'], dtype='float32')
                else:
                    x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
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
    def __init__(self, batch_size, img_paths, p_transform, data_prep_fun, label_prep_fun, rng,
                 random, infinite, full_batch, **kwargs):


        self.img_paths = img_paths
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
                    
                    if 'train' in img_path:
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

class DiscriminatorDataGenerator(object):
    def __init__(self, dataset, batch_size, pos_batch_size, label_id, img_ids, p_transform, data_prep_fun, label_prep_fun, rng,
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
        self.label_id = label_id
        self.pos_batch_size = pos_batch_size
        self.neg_batch_size = batch_size - pos_batch_size
        print np.amax(self.img_ids), np.amin(self.img_ids)  
        selection_labels = self.labels[self.img_ids,:] 
        self.pos_ids = np.where(selection_labels[:,label_id]>0)[0]
        self.neg_ids = np.where(selection_labels[:,label_id]<1)[0]
        print label_id
        print self.pos_ids, self.neg_ids
        
        print len(self.pos_ids), len(self.neg_ids), self.labels.shape[0]
        assert((len(self.pos_ids)+len(self.neg_ids)) == len(img_ids))

    def generate(self):
        while True:
            pos_rand_ids = np.arange(len(self.pos_ids))
            neg_rand_ids = np.arange(len(self.neg_ids))
            neg_ptr = 0
            if self.random:
                self.rng.shuffle(pos_rand_ids)
                self.rng.shuffle(neg_rand_ids)

            for pos_, neg_ in itertools.izip(xrange(0, len(pos_rand_ids), self.pos_batch_size), xrange(0, len(neg_rand_ids), self.neg_batch_size)):
                pos_ids_batch = pos_rand_ids[pos_:pos_ + self.pos_batch_size]
                neg_ids_batch = neg_rand_ids[neg_:neg_ + self.neg_batch_size]
                nb_pos = len(pos_ids_batch)
                nb_neg = len(neg_ids_batch)
                nb = nb_pos + nb_neg

                pos_ids_batch = np.array([self.pos_ids[pid] for pid in pos_ids_batch])
                neg_ids_batch = np.array([self.neg_ids[pid] for pid in neg_ids_batch])

                idxs_batch = np.hstack([pos_ids_batch, neg_ids_batch])

                # allocate batches
                if self.p_transform['channels']:
                    x_batch = np.zeros((nb,self.p_transform['channels'],) + self.p_transform['patch_size'], dtype='float32')
                else:
                    x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
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

class BalancedDataGenerator(object):
    def __init__(self, dataset, batch_size, label_id, img_ids, p_transform, data_prep_fun, label_prep_fun, rng,
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
        self.label_id = label_id
        all_pos_ids, all_neg_ids = app.get_pos_neg_ids(label_id)
        pos_ids = []
        neg_ids = []
        for img_id in img_ids:
            if img_id in all_pos_ids:
                pos_ids.append(img_id)
            elif img_id in all_neg_ids:
                neg_ids.append(img_id)
            else:
                raise
        self.pos_ids = pos_ids
        self.neg_ids = neg_ids
        self.min_samples = min(len(pos_ids), len(neg_ids))

        for img_id in pos_ids:
            if img_id in neg_ids:
                raise

    def generate(self):
        while True:
            rand_pos_idxs = np.arange(len(self.pos_ids))
            rand_neg_idxs = np.arange(len(self.neg_ids))
            if self.random:
                self.rng.shuffle(rand_pos_idxs)
                self.rng.shuffle(rand_neg_idxs)

            min_idxs = min(len(rand_pos_idxs), len(rand_neg_idxs))
            for pos in xrange(0, min_idxs, self.batch_size//2):
                pos_idxs_batch = rand_pos_idxs[pos:pos + self.batch_size//2]
                neg_idxs_batch = rand_neg_idxs[pos:pos + self.batch_size//2]
                idxs_batch = np.concatenate([pos_idxs_batch,neg_idxs_batch])
                nb = len(idxs_batch)
                # allocate batches
                if self.p_transform['channels']:
                    x_batch = np.zeros((nb,self.p_transform['channels'],) + self.p_transform['patch_size'], dtype='float32')
                else:
                    x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
                y_batch = np.zeros((nb,), dtype='float32')

                batch_ids = []

                for i, idx in enumerate(pos_idxs_batch):
                    #print i, idx, self.pos_ids[idx]
                    img_id = self.pos_ids[idx]
                    batch_ids.append(img_id)

                for i, idx in zip(np.arange(len(pos_idxs_batch),len(pos_idxs_batch)+len(neg_idxs_batch)), neg_idxs_batch):
                    #print '-', i, idx, self.neg_ids[idx]
                    img_id = self.neg_ids[idx]
                    batch_ids.append(img_id)


                for i, img_id in enumerate(batch_ids):
                    # print img_id
                    img = app.read_compressed_image(self.dataset, img_id)
                    x_batch[i] = self.data_prep_fun(x=img)
                    y_batch[i] = self.label_prep_fun(self.labels[img_id])


                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break


class TTADataGenerator(object):
    def __init__(self, dataset, tta, img_ids, p_transform, data_prep_fun, label_prep_fun, rng,
                 random, duplicate_label = True, labels=app.get_labels_array(), **kwargs):


        self.dataset = dataset
        self.img_ids = img_ids
        self.nsamples = len(img_ids)
        self.tta = tta
        self.batch_size = tta.n_augmentations
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
        self.label_prep_fun = label_prep_fun
        self.rng = rng
        self.random = random
        self.labels = labels
        self.duplicate_label = duplicate_label

    def generate(self):
        rand_idxs = np.arange(len(self.img_ids))
        if self.random:
            self.rng.shuffle(rand_idxs)
        for pos in xrange(len(rand_idxs)):
            imid = rand_idxs[pos]
            nb = self.batch_size
            # allocate batches
            if self.p_transform['channels']:
                x_batch = np.zeros((nb,self.p_transform['channels'],) + self.p_transform['patch_size'], dtype='float32')
            else:
                x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
            
            if self.p_transform['n_labels']>1:
                y_batch = np.zeros((nb, self.p_transform['n_labels']), dtype='float32')
            else:
                y_batch = np.zeros((nb,), dtype='float32')

            img_id = self.img_ids[imid]
            try:
                img = app.read_compressed_image(self.dataset, img_id)
            except Exception:
                print 'cannot open ', self.dataset, img_id
            x_batch = self.tta.make_augmentations(self.data_prep_fun(x=img))
            if self.duplicate_label:
                y_batch = self.tta.duplicate_label(self.label_prep_fun(self.labels[img_id]))

            yield x_batch, y_batch, img_id

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


def _test_balanced_data_generator():
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

    dg = BalancedDataGenerator(dataset='train',
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
    for (x_chunk, y_chunk, id_train) in dg.generate():
        print x_chunk.shape
        print y_chunk.shape
        print id_train
        print y_chunk


def _test_discriminator_data_generator():
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

    dg = BalancedDataGenerator(dataset='train',
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
    for (x_chunk, y_chunk, id_train) in dg.generate():
        print x_chunk.shape
        print y_chunk.shape
        print id_train
        print y_chunk


if __name__ == "__main__":
    _test_data_generator()


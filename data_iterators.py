import numpy as np


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

                    # print img_id
                    img = app.read_compressed_image(self.dataset, img_id)
                    x_batch[i] = self.data_prep_fun(x=img)
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

def _test_data_generator():
        #testing data iterator 

    p_transform = {'patch_size': (128, 128),
               'channels': 4,
               'n_labels': 17}
    rng = np.random.RandomState(42)

    def data_prep_fun(x):
        x = x[:,:128,:128]
        return x

    folds = app.make_stratified_split(no_folds=5)
    all_ids = folds[0] + folds[1] + folds[2] + folds[3] +folds[4]
    bad_ids = [18772, 28173, 5023]
    img_ids = [x for x in all_ids if x not in bad_ids]

    dg = DataGenerator(dataset='train',
                        batch_size=10,
                        img_ids = img_ids,
                        p_transform=p_transform,
                        data_prep_fun = data_prep_fun,
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


if __name__ == "__main__":
    _test_balanced_data_generator()


import numpy as np


import pathfinder
import utils
import app
import buffering

class DataGenerator(object):
    def __init__(self, dataset, batch_size, img_ids, p_transform, data_prep_fun, rng,
                 random, infinite, full_batch, **kwargs):


        self.dataset = dataset
        self.img_ids = img_ids
        self.nsamples = len(img_ids)
        self.batch_size = batch_size
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
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
                y_batch = np.zeros((nb, self.p_transform['n_labels']), dtype='float32')

                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_id = self.img_ids[idx]
                    batch_ids.append(img_id)

                    # print img_id
                    img = app.read_compressed_image(self.dataset, img_id)
                    x_batch[i] = self.data_prep_fun(x=img)
                    y_batch[i] = self.labels[img_id]


                    #print 'i', i, 'img_id', img_id, y_batch[i]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break


if __name__ == "__main__":
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

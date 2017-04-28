import numpy as np


import pathfinder
import utils
import app

class DataGenerator(object):
    def __init__(self, dataset, batch_size, img_ids, p_transform, data_prep_fun, rng,
                 random, infinite, **kwargs):


        self.dataset = dataset
        self.img_ids = img_ids
        self.batch_size = batch_size
        self.p_transform = p_transform
        self.data_prep_fun = data_prep_fun
        self.rng = rng
        self.random = random
        self.infinite = infinite

        self.labels = app.get_labels()

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
                    x_batch = np.zeros((nb,) + self.p_transform['patch_size'], dtype='float32')
                else:
                    x_batch = np.zeros((nb, p_transform['channels']) + self.p_transform['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, self.p_transform['n_labels']), dtype='float32')

                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_id = self.img_ids[idx]
                    batch_ids.append(img_id)

                    img = app.read_image(self.dataset, img_id)
                    x_batch[i] = self.data_prep_fun(data=img)
                    
                    y_batch[i] = self.labels[img_id]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break

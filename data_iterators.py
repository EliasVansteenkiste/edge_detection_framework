import numpy as np

import app
import buffering


class EdgeDataGenerator(object):
    def __init__(self, mode, batch_size, img_id_pairs, data_prep_fun, edges_prep_fun, rng,
                 random, infinite, full_batch, **kwargs):

        self.mode = mode
        self.img_id_pairs = img_id_pairs
        self.nsamples = len(img_id_pairs)
        self.batch_size = batch_size
        self.data_prep_fun = data_prep_fun
        self.edges_prep_fun = edges_prep_fun
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.full_batch = full_batch

    def generate(self):
        while True:
            rand_idxs = np.arange(len(self.img_id_pairs))
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in range(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)

                # allocate batches
                x_batch = []
                y_batch = []
                batch_ids = []

                for i, idx in enumerate(idxs_batch):
                    img_id_pair = self.img_id_pairs[idx]
                    batch_ids.append(img_id_pair)
                    img_id, edges_id = img_id_pair
                    try:
                        img = app.read_image_from_path(img_id)
                        edges = app.read_image_from_path(edges_id)
                    except Exception:
                       ` print('cannot open ', img_id, edges_id)

                    x = self.data_prep_fun(x=img)
                    y = self.edges_prep_fun(edges=edges)
                    x_batch.append(x)
                    if 'train' in self.mode:
                        y_batch.append(y)

                    print('i', i, 'img_id', img_id_pair)

                x_batch = np.stack(x_batch)
                y_batch = np.stack(y_batch)
                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, batch_ids
                else:
                    yield x_batch, y_batch, batch_ids

            if not self.infinite:
                break


def _test_data_generator():
    # testing data iterator
    rng = np.random.RandomState(42)

    def data_prep_fun(x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        return x

    def edges_prep_fun(edges):
        edges = edges[None, :, :]
        return edges

    img_id_pairs = app.get_id_pairs('test_data/test1/trainA', 'test_data/test1_hed/trainA')

    dg = EdgeDataGenerator(mode='train',
                           batch_size=2,
                           img_id_pairs=img_id_pairs,
                           data_prep_fun=data_prep_fun,
                           edges_prep_fun=edges_prep_fun,
                           rng=rng,
                           full_batch=False, random=False, infinite=False)

    for (x_chunk, y_chunk, id_train) in buffering.buffered_gen_threaded(dg.generate()):
        print(x_chunk.shape, y_chunk.shape, id_train)


if __name__ == "__main__":
    _test_data_generator()

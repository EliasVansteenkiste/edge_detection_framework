import numpy as np # linear algebra
import scipy
import scipy.io as io
from PIL import Image
import os

import pathfinder
import utils
# import utils_plots

rng = np.random.RandomState(37145)


# def read_mat(dataset, idx, plot=False):
#     path = pathfinder.DATA_PATH + '/' + dataset + '/' + str(idx) + '.mat'
#     d_mat = io.loadmat(path)
#     print(d_mat['groundTruth'].shape)
#     fmaps = []
#     for midx, gmap in enumerate(d_mat['groundTruth'][0]):
#         if plot:
#             utils_plots.show_img(gmap[0][0][0], dataset+'_'+str(idx)+'_'+str(midx))
#         fmaps.append(gmap[0][0][0])
#     return np.stack(fmaps)



def read_image(dataset, filename):
    path = pathfinder.DATA_PATH + '/' + dataset + '/' + filename
    im = Image.open(path)
    arr = np.asanyarray(im)
    return arr


def read_image_from_path(path):
    im = Image.open(path)
    arr = np.asanyarray(im)
    return arr


def read_image_from_id(id):
    path = pathfinder.DATA_PATH + '/' + id
    im = Image.open(path)
    arr = np.asanyarray(im)
    return arr


def get_id_pairs(dataset_img, dataset_edges):
    img_path = os.path.join(os.sep, pathfinder.DATA_PATH, dataset_img)
    edges_path = os.path.join(os.sep, pathfinder.DATA_PATH, dataset_edges)

    filenames = os.listdir(img_path)
    id_pairs = []
    for filename in filenames:
        if filename[0] != '.':
            img_filename = os.path.join(os.sep, img_path, filename)
            edges_filename = os.path.join(os.sep, edges_path, filename)
            assert os.path.exists(edges_filename), 'corresponding edge map not found: '+edges_filename
            id_pairs.append((img_filename, edges_filename))
    return id_pairs

if __name__ == "__main__":
    read_image('test_data', 'test1/trainA/20170831-22-46-33_0000000004.jpg')
    read_image('test_data', 'test1_hed/trainA/20170831-22-46-33_0000000004.jpg')
    print(get_id_pairs('test_data/test1/trainA', 'test_data/test1_hed/trainA'))





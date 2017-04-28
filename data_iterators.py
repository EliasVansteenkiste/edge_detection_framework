import numpy as np


import pathfinder
import utils
import app

class LunaDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, rng,
                 random, infinite, patient_ids=None, **kwargs):

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                  pixel_spacing=pixel_spacing,
                                                                  luna_annotations=
                                                                  self.id2annotations[pid],
                                                                  luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]

                yield x, y, None, annotations, tf_matrix, pid

            if not self.infinite:
                break

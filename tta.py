import string
import numpy as np

import app
import data_transforms


class TTA():
    n_augmentations = 0

    def duplicate_label(self, label):
        assert self.n_augmentations > 0
        duplication = []
        for i in range(self.n_augmentations):
            duplication.append(np.copy(label))
        return np.float32(np.stack(duplication))


class LosslessTTA(TTA):

    def __init__(self, p_aug):
        self.p_aug = p_aug
        self.n_augmentations = len(p_aug['rot90_values']) * len(p_aug['flip'])

    def make_augmentations(self, original_img):
        return data_transforms.generate_all_lossless(original_img, self.p_aug)

class LossLessPlusTranslation(TTA):
    def __init__(self, p_aug, p_trans):
        self.p_aug = p_aug
        self.p_trans = p_trans
        self.n_augmentations = len(p_aug['rot90_values']) * len(p_aug['flip'])*3

    def make_augmentations(self, original_img):
        return data_transforms.generate_all_lossless_plus_translation(original_img, self.p_aug,self.p_trans)
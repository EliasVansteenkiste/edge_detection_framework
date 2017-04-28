from collections import namedtuple
import numpy as np
import scipy.ndimage
import math
import skimage.io
import skimage.transform
from collections import defaultdict

import utils_plots
import app


rng = np.random.RandomState(37145)


def pixelnorm(x, MIN=0, MAX=61440.0):
    x = (x - MIN) / (MAX - MIN)
    return x



default_channel_zmuv_stats = {
    'avg': [4970.55, 4245.35, 3064.64, 6360.08],
    'std': [1785.79, 1576.31, 1661.19, 1841.09]}

def channel_zmuv(x, img_stats = default_channel_zmuv_stats, no_channels=4):
    for ch in range(no_channels):
        x[ch] = (x[ch] - img_stats['avg'][ch]) / img_stats['std'][ch]
    return x

default_channel_norm_stats = {
    0.1: [2739., 2022., 1284., 1091.],
    0.5: [3016., 2272., 1433., 1415.],
    1: [3149., 2441., 1563.,  1733.],
    5: [3514., 2867., 1792., 3172.],
    10: [3661., 3016., 1902., 4132.],
    50: [4503., 3768., 2534., 6399.],
    90: [6615., 5912., 4694., 8311.],
    95: [7623., 6822., 5698., 9109.],
    99: [11065., 10184., 9047., 11561.],
    99.5: [14686., 13508., 12197., 12820.],
    99.9: [23722., 16926., 19183., 16523.]
}

def channel_norm(x, img_stats = default_channel_norm_stats, percentiles=[1,99], no_channels=4):
    for ch in range(no_channels):
        minimum = img_stats[percentiles[0]][ch]
        maximum = img_stats[percentiles[1]][ch]
        x[ch] = (x[ch] - minimum) / (maximum-minimum)

default_augmentation_params = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter

def build_centering_transform(image_shape, target_shape=(50, 50)):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))

def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)

def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0) # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False): 
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment

def perturb(img, augmentation_params, target_shape, rng=rng, n_channels=4):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    img_spat_shape = (img.shape[1], img.shape[2])
    tform_centering = build_centering_transform(img_spat_shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(img_spat_shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    
    chs = []
    for ch in range(n_channels):
        print img[ch].shape
        ch_warped = fast_warp(img[ch], tform_centering + tform_augment, output_shape=target_shape, mode='constant').astype('float32')
        print ch_warped.shape
        chs.append(ch_warped)
    out_img = np.stack(chs, axis=0)
    return out_img



def _print_stats_channels(img, channel_stats, channel_data):
    n_channels = img.shape[-1]
    print n_channels
    for ch in range(n_channels):
        print 'ch', ch,
        ch_data = img[:,:,ch]
        channel_data[ch].append(img[:,:,ch])
        print 'max', np.amax(ch_data),
        channel_stats[str(ch)+'max'].append(np.amax(ch_data))
        print 'min', np.amin(ch_data),
        channel_stats[str(ch)+'min'].append(np.amin(ch_data))
        print 'avg', np.average(ch_data), 
        channel_stats[str(ch)+'avg'].append(np.average(ch_data))
        print 'std', np.std(ch_data),
        channel_stats[str(ch)+'std'].append(np.std(ch_data))
        print 'var', np.var(ch_data)
        channel_stats[str(ch)+'var'].append(np.var(ch_data))


if __name__ == "__main__":
    channel_stats = defaultdict(list)
    channel_data = defaultdict(list)
    for i in range(2000):
        #read in image
        print '***** ',i 
        tif = app.read_image('train', i)
        print tif.shape
        utils_plots.plot_img(tif,'plots/test'+str(i)+'.jpg')
        p_tif = perturb(tif, default_augmentation_params, (256,256), rng)
        utils_plots.plot_img(p_tif,'plots/test'+str(i)+'_random_augm.jpg')


        tif = tif.astype('float32')
        print tif.shape
        print tif.dtype
        #_print_stats_channels(tif,channel_stats,channel_data)

    # print 'overall stats'


    # for ch in range(4):
    #     print 'ch', ch,
    #     ch_data = np.concatenate(channel_data[ch])
    #     for p in [0.1,0.5,1,5,10,50,90,95,99,99.5,99.9]:
    #         print p, '-', np.percentile(ch_data,p), '|',
    #     print 
    #     print 'avg', np.average(ch_data)
    #     print 'std', np.std(ch_data)



        #utils_plots.show_img(calibrate_image(tif[:,:,:3]))


    
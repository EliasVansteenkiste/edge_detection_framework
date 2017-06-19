import numpy as np

import data_iterators
import app


def copy_imgs(img_ids, target_path):

    p_transform = {'patch_size': (256, 256),
               'channels': 4,
               'n_labels': 17}

    rng = np.random.RandomState(42)

    def data_prep_fun(x):
        return x

    def label_prep_fun(labels):
        return labels

    dg = data_iterators.SlimDataGenerator(dataset='train-jpg',
                        batch_size=1,
                        img_ids = img_ids,
                        p_transform=p_transform,
                        data_prep_fun = data_prep_fun,
                        label_prep_fun = label_prep_fun,
                        rng=rng,
                        full_batch=False, random=False, infinite=False)

    for idx, (x_chunk, y_chunk, id_train) in enumerate(dg.generate()):
        if idx % 100 == 0:
            print str(idx), 'processed'
        img = x_chunk[0]
        img_id = id_train[0]
        rgb_img = img.convert('RGB')
        rgb_img.save(target_path+'/train_'+str(img_id)+'.jpg')



if __name__ == "__main__":
    #ids_fn_cultivation = [17858, 21910, 2970, 5432, 15438, 16352, 5174, 25700, 25987, 29864, 34600, 35058, 35595, 36273, 36702, 36775, 37506, 38041, 38078, 38703, 38861, 40258, 23778, 28200, 30466, 28002, 28165, 29234, 29313, 30000, 30482, 30483, 30691, 31666, 30897, 38327, 35141, 11908, 1536, 3535, 3702, 5213, 6318, 2278, 4855, 5101, 5345, 5611, 6304, 33052, 39936, 39135, 8826, 9011, 9212, 9263, 9556, 9834, 11762, 11942, 12242, 12455, 12810, 13157, 13297, 13585, 13863, 13994, 14302, 14652, 14711, 15557, 1066, 1154, 1407, 2143, 2250, 2317, 2322, 2809, 3444, 4093, 5408, 5615, 6504, 7540, 9702, 9864, 10037, 10346, 10633, 10735, 11337, 11433, 11691, 11693, 11912, 11996, 12238, 12289, 12459, 12576, 12662, 12766, 13026, 13275, 13499, 13764, 13976, 14133, 14636, 15006, 15321, 15636, 16645, 16772, 18820, 19827, 20220, 20272, 21327, 21337, 22388, 23659, 24055, 17167, 23467, 24185, 25507, 25691, 26967, 27975, 28044, 28790, 29711, 30127, 30263, 1061, 2043, 19992, 22213, 3204, 30363, 9110, 12350, 7288, 26909, 11321, 17278, 32345, 32525, 32705, 17499, 20863, 21612, 18249, 30820, 951, 27052, 27993, 29327, 29612, 30656, 32990, 33409, 33697, 3332, 3556, 5799, 5904, 7290, 31685, 34326, 34620, 35672, 36619, 37636, 37664, 38148, 38846, 38871, 39532, 39772, 40392, 5976, 34938, 35009, 8448, 10384, 31142, 33550, 35053, 12437, 24757, 24960, 25142, 25147, 25733, 25750, 27124, 29006, 29037, 29303, 29545, 29800, 29866, 30388, 31127, 31314, 32306, 32399, 8486, 9254, 9965, 10811, 12355, 12430, 12593, 15287, 15338, 16158, 17577, 16244, 16299, 16950, 17950, 18169, 18309, 18560, 18576, 18700, 19248, 19695, 19759, 20739, 20750, 21249, 21620, 21751, 22207, 22407, 22409, 22737, 22936, 23037, 23181, 23462, 23716, 24321, 26707, 27398, 29962, 30774, 26454, 29886, 31407, 33031, 33848, 34527, 34582, 35036, 35352, 35604, 35716, 36686, 39249, 16676, 18194, 20300, 20667, 21295, 8390, 17127, 35451, 185, 2667, 39320, 3137, 33649, 38272, 14453, 14783, 38754, 19270, 24128, 10277, 10915, 11236, 12002, 12432, 16047, 23649, 21949, 24193, 28084, 3113, 5566, 5740, 15980, 17008, 17255, 17352, 18755, 18926, 21362, 31497, 32075, 32892, 35856, 19264, 19591, 20221, 21601, 22262, 23453, 23562, 24519, 25993, 26092, 26181, 26434, 7779, 10842, 13229, 18333, 23570, 24333, 24978, 25938, 15528, 21503, 19381, 27144, 27197, 30601, 16335, 17633, 22003, 22155, 3997, 8148, 37223, 17206, 36386, 19135, 32781, 33486, 33717, 34171, 35401, 35593, 35864, 36438, 38134, 39241, 24034, 25823, 26119, 28118, 28123, 28812, 30176, 30753, 30855, 30953, 32146, 32639, 32851, 33207, 33599, 33660, 33683, 33701, 34031, 34487, 35024, 35348, 35504, 35540, 35634, 35792, 36024, 36235, 37155, 37177, 37257, 37317, 37493, 38215, 38923, 39027, 39101, 39530, 39799, 39887, 39948, 40354, 40389, 2894, 6037, 9041, 10253, 34666, 17338, 18107, 18109, 18471, 18514, 18711, 19868, 20453, 20862, 20984, 21299, 22287, 23452, 37828, 38844, 39194, 39475, 36624, 10137, 7582, 19904, 24651, 25260, 13316, 33540, 37242, 38284, 6820, 14594, 17013, 30575, 32469, 34953, 1840, 21373, 23279, 34128, 15457, 1317, 1747, 1860, 2001, 2480, 2748, 4029, 5492, 6069, 6704, 7802, 7561, 8803, 9450, 10622, 10777, 998, 1129, 1497, 2549, 5064, 5374, 5963, 6603, 7496, 8310, 8441, 8512, 15470, 16932, 19100, 385, 2083, 4117, 1691, 39615, 39940, 27650, 28314, 28971, 12121, 13319, 35865, 28387, 32386, 567, 816, 1167, 1271, 1554, 1951, 2155, 2716, 2758, 3123, 4194, 4319, 4611, 4962, 5125, 5461, 5634, 5841, 6132, 6516, 7268, 7375, 7445, 7651, 7683, 8273, 33293, 33372, 33690, 34749, 35514, 35722, 35918, 37123, 37459, 38452, 25085, 25928, 25942, 26394, 27054, 27434, 27533, 27925, 28350, 28430, 28869, 28958, 29164, 29307, 29416, 29728, 29999, 30693, 31270, 31655, 31718, 11022, 11922, 12970, 13672, 14931, 15084, 3885, 12369, 12742, 25188, 29954, 225, 613, 770, 949, 1234, 1599, 2051, 2244, 3054, 4911, 5369, 5459, 5624, 5791, 7736, 7777, 7808, 24930, 26824, 26847, 29758, 29810, 30631, 25918, 23642, 9803, 31918, 27454, 9528, 26487]   
    #ids_fn_water = [36915, 37497, 38990, 39279, 39984, 15779, 24623, 26018, 26148, 6427, 3969, 5174, 16331, 16351, 17738, 17764, 19836, 20031, 20882, 21052, 21214, 22238, 23527, 23927, 24279, 23102, 24460, 25678, 27429, 27783, 441, 1207, 3459, 5154, 6134, 6332, 38435, 28820, 30294, 33390, 21724, 30466, 30897, 32389, 38327, 39964, 9293, 33827, 35669, 37337, 38159, 17926, 18048, 22958, 416, 7480, 7881, 5112, 39774, 12196, 18067, 25462, 30154, 32081, 23530, 23680, 23944, 24054, 24552, 24629, 25303, 26032, 26691, 27361, 27557, 27646, 27755, 27782, 27829, 28070, 28445, 28543, 28994, 29210, 29282, 29803, 29959, 30111, 30515, 30688, 30920, 31466, 31564, 22521, 16295, 16340, 17169, 17492, 17497, 19905, 19944, 19980, 21221, 21322, 21384, 21734, 22547, 22676, 22816, 23148, 1407, 1499, 2143, 3906, 4204, 5158, 5304, 6110, 6496, 6504, 7039, 7667, 222, 680, 1061, 2043, 3437, 3553, 4183, 5416, 26301, 19482, 34751, 21869, 25133, 15188, 6368, 19683, 26757, 29839, 29997, 30191, 32048, 32352, 28311, 12633, 283, 614, 928, 2869, 3636, 4439, 5169, 5372, 5872, 6104, 6108, 6220, 34064, 35554, 15050, 15447, 17913, 21450, 32184, 32703, 33334, 33517, 34421, 35436, 37787, 34005, 8770, 10594, 17058, 3637, 5319, 5904, 4644, 35081, 34826, 40238, 24515, 24681, 25066, 25193, 25478, 26101, 26595, 27619, 27863, 28390, 28627, 28634, 28667, 29422, 30216, 30612, 31381, 31464, 31620, 31745, 31801, 32466, 31107, 31654, 32393, 3185, 4672, 28760, 31393, 31604, 32673, 34304, 38198, 38986, 39624, 40142, 158, 324, 568, 1671, 1843, 2240, 2409, 2541, 3757, 3908, 4107, 4346, 6268, 6336, 6397, 6415, 7215, 7225, 7927, 8040, 8131, 16877, 16976, 17092, 17311, 17446, 19060, 19124, 19137, 19214, 19401, 19758, 19783, 19839, 20700, 20869, 20971, 21262, 21309, 22233, 22510, 22533, 22729, 22777, 22784, 22883, 23283, 23341, 23508, 14379, 16143, 31313, 31656, 32369, 32654, 33544, 33930, 36000, 36334, 38832, 39160, 39670, 40223, 40259, 40413, 9254, 10545, 12355, 12541, 13541, 15287, 25876, 26707, 27101, 28076, 29202, 30300, 35491, 38719, 20943, 21295, 19055, 31512, 36613, 4960, 6990, 8902, 24033, 32646, 35636, 36169, 20299, 24199, 3113, 35704, 36223, 37057, 37460, 38239, 39431, 39856, 4436, 34635, 4549, 5740, 5322, 9722, 9956, 12206, 12971, 13031, 14126, 17621, 18119, 18120, 18743, 19075, 19274, 19909, 30084, 33726, 34449, 34785, 34942, 34773, 39341, 39980, 10842, 21503, 27807, 11, 941, 1792, 2522, 3118, 5308, 5505, 5778, 6932, 13063, 15234, 10495, 11287, 25893, 27537, 14382, 28523, 7088, 9895, 9973, 10454, 10508, 11231, 11490, 11795, 13556, 13841, 13935, 14051, 14732, 32391, 32848, 33107, 34094, 34490, 34852, 35306, 35490, 35498, 35547, 36069, 36218, 36528, 36550, 36558, 36639, 37267, 38204, 38646, 38775, 39452, 39682, 39785, 52, 651, 977, 1098, 1790, 2807, 3068, 3285, 5065, 5426, 5679, 7709, 7838, 8004, 8376, 25900, 26602, 31608, 574, 665, 670, 4499, 5036, 33989, 34927, 37828, 24317, 13409, 20592, 24491, 3639, 2595, 3931, 12451, 8431, 9368, 9401, 10307, 10587, 11122, 11304, 12410, 13941, 13695, 474, 2281, 3267, 4440, 5051, 9501, 10558, 11907, 12934, 14106, 19658, 20528, 27853, 8708, 8832, 10974, 11564, 19100, 1691, 21381, 21576, 16226, 16577, 16982, 17203, 17375, 17595, 17647, 18817, 20376, 20956, 21240, 22653, 6730, 23776, 15934, 18277, 29010, 29394, 1012, 13975, 16394, 7157, 16894, 17072, 17650, 17998, 18144, 18584, 18877, 19857, 20118, 20716, 20857, 21198, 21652, 22363, 23346, 23396, 301, 391, 411, 525, 565, 632, 722, 872, 1388, 1393, 1918, 2067, 2293, 2315, 2460, 3374, 3753, 3961, 4011, 4184, 4278, 4599, 4631, 4652, 6211, 6418, 6879, 7275, 7661, 7918, 8010, 8076, 8140, 8248, 27184, 27268, 27308, 28327, 28783, 28817, 23735, 24049, 24133, 24561, 25378, 25503, 25515, 25697, 25841, 25930, 27360, 27708, 30833, 31011, 33132, 35162, 35897, 13019, 13672, 15084, 15095, 16323, 24660, 26824, 30835, 31430, 26576, 28622, 29249, 12772, 18986, 13931, 31469, 1465, 10924]
    
    #copy_imgs(app.get_ids_by_tag(13), 'manual_labeling/habitation')
    copy_imgs(app.get_ids_by_tag(8), 'manual_labeling/blooming')
    # copy_imgs(ids_fn_cultivation, 'manual_labeling/fn_cultivation')
    # copy_imgs(ids_fn_water, 'manual_labeling/fn_water')
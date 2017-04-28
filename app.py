import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage

import pathfinder

rng = np.random.RandomState(37145)



def read_image(dataset, id):
	if dataset == 'train':
		prefix = 'train-tif/train_'
	elif dataset == 'test':
		prefix = 'test-tif/test_'
	else:
		raise
	path = pathfinder.DATA_PATH + prefix + str(id) + '.tif'
	image = skimage.io.imread(path)
	image = np.swapaxes(image,0,2)
	return image
	

def get_labels():
	df_train = pd.read_csv(pathfinder.DATA_PATH+'train.csv')
	df_train = pd.concat([df_train['image_name'], df_train.tags.str.get_dummies(sep=' ')], axis=1)
	return df_train

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def top_occ(feat_comb, n_top = 5):
	# a method for printing top occurences of feature combinations
	# built for checking if split is stratified
	n_total_samples = len(feat_comb)
	feat_comb_occ = np.bincount(feat_comb)
	top = feat_comb_occ.argsort()[-n_top:][::-1]
	for idx, fc in enumerate(top):
		print idx, fc, 1.0*feat_comb_occ[fc]/n_total_samples
	print 'checksum', sum(feat_comb)

def make_stratified_split(no_folds=5, verbose=False):
	df = get_labels()
	only_labels = df.drop(['image_name'], axis = 1, inplace = False)
	only_labels = only_labels.as_matrix()
	if verbose: print 'labels shape', only_labels.shape
	feat_comb = only_labels.dot(1 << np.arange(only_labels.shape[-1] - 1, -1, -1))
	feat_comb_set = set(feat_comb)
	feat_comb_occ = np.bincount(feat_comb)
	feat_comb_high = np.where(feat_comb_occ >= no_folds)[0]
	n_total_samples = 0
	folds = [[] for _ in range(no_folds)]
	for fc in feat_comb_high:
		idcs = np.where(feat_comb == fc)[0]
		chunks = chunkIt(idcs,no_folds)
		# print len(idcs), [len(chunk) for chunk in chunks]
		rng.shuffle(chunks)
		for idx, chunk in enumerate(chunks):
			folds[idx].extend(chunk)

	feat_comb_low = np.where(np.logical_and(feat_comb_occ < no_folds, feat_comb_occ > 0))[0]
	low_idcs = []
	for fc in feat_comb_low:
		idcs = np.where(feat_comb == fc)[0]
		low_idcs.extend(idcs)

	chunks = chunkIt(low_idcs,no_folds)
	rng.shuffle(chunks)
	for idx, chunk in enumerate(chunks):
		folds[idx].extend(chunk)


	n_samples_fold = 0
	for f in folds:
		n_samples_fold += len(f)

	if verbose:
		print 'n_samples_fold', n_samples_fold
		top_occ(feat_comb)
		for f in folds:
			top_occ(feat_comb[f])

	return folds





if __name__ == "__main__":
	make_stratified_split(verbose=True)


import string
import sys

import numpy as np

import sklearn
from datetime import datetime

import buffering
import pathfinder
import utils
from configuration import config, set_configuration
import logger
import app
import torch
import os
import cPickle
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description='Evaluate dataset on trained model.')

save_dir = "../data/temp/"

parser.add_argument("config_name", type=str, help="Config name")
parser.add_argument("eval", type=str, help="test/valid/feat/train/test_tta/valid_tta")
parser.add_argument("--dump", type=int, default=0, help="Should we store the predictions in raw format")
parser.add_argument("--best", type=int, default=0, help="Should we use the best model instead of the last model")
args = parser.parse_args()

config_name = args.config_name
set_configuration('configs', config_name)

all_tta_feat = args.eval == 'all_tta_feat'
feat = args.eval == 'feat'

train = args.eval == 'train'
train_tta = args.eval == 'train_tta'
train_tta_feat = args.eval == 'train_tta_feat'

valid = args.eval == 'valid'
valid_tta = args.eval == 'valid_tta'
valid_tta_feat = args.eval == 'valid_tta_feat'
valid_tta_majority = args.eval == 'valid_tta_majority'

test = args.eval == 'test'
test_tta = args.eval == 'test_tta'
test_tta_feat = args.eval == 'test_tta_feat'
test_tta_majority = args.eval == 'test_tta_majority'

dump = args.dump
best = args.best

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name, best=best)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

if best:
    expid += "-best"

print("logs")
# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout
print("prediction path")
# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid

if valid_tta_feat or test_tta_feat or all_tta_feat or train_tta_feat:
    outputs_path += '/features'

utils.auto_make_dir(outputs_path)

if dump:
    prediction_dump = os.path.join(outputs_path, expid + "_" + args.eval + "_predictions.p")

print('Build model')
model = config().build_model()
model.l_out.load_state_dict(metadata['param_values'])
model.l_out.cuda()
model.l_out.eval()
criterion = config().build_objective()

if test:
    data_iterator = config().test_data_iterator
elif feat:
    data_iterator = config().feat_data_iterator


def get_preds_targs(data_iterator):
    print('Data')
    print('n', sys.argv[2], ': %d' % data_iterator.nsamples)

    validation_losses = []
    preds = []
    targs = []
    ids = []

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):

        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(), volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(), volatile=True)

        predictions = model.l_out(inputs)
        loss = criterion(predictions, labels)
        validation_losses.append(loss.cpu().data.numpy()[0])
        targs.append(y_chunk)
        if feat:
            for idx, img_id in enumerate(id_chunk):
                np.savez(open(outputs_path + '/' + str(img_id) + '.npz', 'w'), features=predictions[idx])

        preds.append(predictions.cpu().data.numpy())
        # print id_chunk, targets, loss
        if n % 50 == 0:
            print(n, 'batches processed')

        ids.append(id_chunk)

    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    ids = np.stack(ids)
    print('Validation loss', np.mean(validation_losses))

    return preds, targs, ids


def get_preds_targs_tta(data_iterator, aggregation="mean", threshold=0.5):
    print('Data')
    print('n', sys.argv[2], ': %d' % data_iterator.nsamples)

    # validation_losses = []
    preds = []
    targs = []
    ids = []

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):
        # load chunk to GPU
        # if n == 10:
        #    break
        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(), volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(), volatile=True)
        predictions = model.l_out(inputs)

        predictions = predictions.cpu().data.numpy()

        if aggregation == "majority":
            final_prediction = np.zeros((predictions.shape[1],))
            for dim in range(predictions.shape[1]):
                count = np.bincount(predictions[:, dim] > threshold, minlength=2)
                final_prediction[dim] = 1 if count[1] >= predictions.shape[0] / 2.0 else 0


        elif aggregation == "mean":
            final_prediction = np.mean(predictions, axis=0)
        # avg_loss = np.mean(loss, axis=0)

        # validation_losses.append(avg_loss)
        targs.append(y_chunk[0])
        ids.append(id_chunk)
        preds.append(final_prediction)

        if n % 1000 == 0:
            print(n, 'batches processed')

    preds = np.stack(preds)
    targs = np.stack(targs)
    ids = np.stack(ids)

    # print 'Validation loss', np.mean(validation_losses)

    return preds, targs, ids


def get_preds_targs_tta_feat(data_iterator, prelabel=''):
    print('Data')
    print('n', sys.argv[2], ': %d' % data_iterator.nsamples)

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):
        # load chunk to GPU
        # if n == 10:
        #    break
        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(), volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(), volatile=True)
        predictions = model.l_out(inputs, feat=True)

        predictions = predictions.cpu().data.numpy()

        # final_prediction = np.mean(predictions.cpu().data.numpy(), axis=0)
        # avg_loss = np.mean(loss, axis=0)

        # validation_losses.append(avg_loss)

        # print(predictions.shape)
        # print(id_chunk)

        for i in range(predictions.shape[0]):
            file = open(os.path.join(outputs_path, prelabel + str(id_chunk) + "_" + str(i) + ".npy"), "wb")
            np.save(file, predictions[i])
            file.close()

        if n % 1000 == 0:
            print(n, 'batches processed')


if train_tta_feat:
    train_it = config().tta_train_data_iterator
    get_preds_targs_tta_feat(train_it)

if all_tta_feat:
    all_it = config().tta_all_data_iterator
    get_preds_targs_tta_feat(all_it)

if train or train_tta:
    if train:
        train_it = config().trainset_valid_data_iterator
        preds, targs, ids = get_preds_targs(train_it)
    elif train_tta:
        train_it = config().tta_train_data_iterator
        preds, targs, ids = get_preds_targs_tta(train_it)
    if dump:
        file = open(prediction_dump, "wb")
        cPickle.dump([preds, targs, ids], file)
        file.close()

if valid_tta_feat:
    valid_it = config().tta_valid_data_iterator
    get_preds_targs_tta_feat(valid_it)

if valid or valid_tta or valid_tta_majority:

    if valid:
        valid_it = config().valid_data_iterator
        preds, targs, ids = get_preds_targs(valid_it)
    elif valid_tta:
        valid_it = config().tta_valid_data_iterator
        preds, targs, ids = get_preds_targs_tta(valid_it)
    elif valid_tta_majority:
        valid_it = config().tta_valid_data_iterator
        preds, targs, ids = get_preds_targs_tta(valid_it, aggregation="majority", threshold=0.53)

    if dump:
        file = open(prediction_dump, "wb")
        cPickle.dump([preds, targs, ids], file)
        file.close()

    tps = [np.sum(qpreds[:, i] * targs[:, i]) for i in range(17)]
    fps = [np.sum(qpreds[:, i] * (1 - targs[:, i])) for i in range(17)]
    fns = [np.sum((1 - qpreds[:, i]) * targs[:, i]) for i in range(17)]

    print('TP')
    print(np.int32(tps))
    print('FP')
    print(np.int32(fps))
    print('FN')
    print(np.int32(fns))

    print('worst classes')
    print(app.get_headers())
    print(4 * np.array(fps) + np.array(fns))

if test_tta_feat:
    test_it = config().tta_test_data_iterator
    get_preds_targs_tta_feat(test_it, prelabel='test_')

    test2_it = config().tta_test2_data_iterator
    get_preds_targs_tta_feat(test2_it, prelabel='file_')

if test or test_tta or test_tta_majority:

    imgid2pred = {}
    imgid2raw = {}

    if test:
        test_it = config().test_data_iterator
        preds, _, ids = get_preds_targs(test_it)
    elif test_tta:
        test_it = config().tta_test_data_iterator
        preds, _, ids = get_preds_targs_tta(test_it)
    elif test_tta_majority:
        test_it = config().tta_test_data_iterator
        preds, _, ids = get_preds_targs_tta(test_it, aggregation="majority")

    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    outputs_path = predictions_dir + '/' + expid
    utils.auto_make_dir(outputs_path)

    output_pickle_file = outputs_path + '/%s-%s.pkl' % (expid, args.eval)
    file = open(output_pickle_file, "wb")
    cPickle.dump(imgid2raw, file)
    file.close()

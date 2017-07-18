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
import submission
import torch
import os
import cPickle
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='Evaluate dataset on trained model.')

save_dir="../data/temp/"

# word level specs
parser.add_argument("config_name",type = str,  help="Config name")
parser.add_argument("eval",type = str,  help="test/valid/feat/train/test_tta/valid_tta")
parser.add_argument("--dump",type = int, default = 0, help="Should we store the predictions in raw format")
parser.add_argument("--best",type = int, default = 0, help="Should we use the best model instead of the last model")
args = parser.parse_args()


config_name = args.config_name
set_configuration('configs_pytorch', config_name)


valid = args.eval =='valid'
test = args.eval == 'test'
feat = args.eval == 'feat'
train = args.eval == 'train'
train_tta = args.eval == 'train_tta'
valid_tta = args.eval == 'valid_tta'
test_tta = args.eval == 'test_tta'
train_tta_feat = args.eval == 'train_tta_feat'
valid_tta_feat = args.eval == 'valid_tta_feat'
test_tta_feat = args.eval == 'test_tta_feat'

dump = args.dump

best = args.best


# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name,best=best)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

if best:
    expid+="-best"

print("logs")
# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout
print("prediction path")
# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid

if valid_tta_feat or test_tta_feat:
    outputs_path += '/features'

utils.auto_make_dir(outputs_path)

if dump:
    prediction_dump = os.path.join(outputs_path,expid+"_"+args.eval+"_predictions.p")

print 'Build model'
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
    print 'Data'
    print 'n', sys.argv[2], ': %d' % data_iterator.nsamples

    validation_losses = []
    preds = []
    targs = []
    ids = []

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):

        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(),volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(),volatile=True)

        predictions = model.l_out(inputs)
        loss = criterion(predictions, labels)
        validation_losses.append(loss.cpu().data.numpy()[0])
        targs.append(y_chunk)
        if feat:
            for idx, img_id in enumerate(id_chunk):
                np.savez(open(outputs_path+'/'+str(img_id)+'.npz', 'w') , features = predictions[idx])

        preds.append(predictions.cpu().data.numpy())
        #print id_chunk, targets, loss
        if n % 50 ==0:
            print n, 'batches processed'

        ids.append(id_chunk)

    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    ids = np.stack(ids)
    print 'Validation loss', np.mean(validation_losses)

    return preds, targs, ids


def get_preds_targs_tta(data_iterator):
    print 'Data'
    print 'n', sys.argv[2], ': %d' % data_iterator.nsamples

    # validation_losses = []
    preds = []
    targs = []
    ids = []

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):
        # load chunk to GPU
        # if n == 10:
        #    break
        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(),volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(),volatile=True)
        predictions = model.l_out(inputs)

        final_prediction = np.mean(predictions.cpu().data.numpy(), axis=0)
        # avg_loss = np.mean(loss, axis=0)

        # validation_losses.append(avg_loss)
        targs.append(y_chunk[0])
        ids.append(id_chunk)
        preds.append(final_prediction)

        if n % 1000 == 0:
            print n, 'batches processed'

    preds = np.stack(preds)
    targs = np.stack(targs)
    ids = np.stack(ids)

    print preds.shape
    print targs.shape
    print ids.shape

    # print 'Validation loss', np.mean(validation_losses)

    return preds, targs, ids

def get_preds_targs_tta_feat(data_iterator,prelabel=''):
    print 'Data'
    print 'n', sys.argv[2], ': %d' % data_iterator.nsamples


    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):
        # load chunk to GPU
        # if n == 10:
        #    break
        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda(),volatile=True), Variable(
            torch.from_numpy(y_chunk).cuda(),volatile=True)
        predictions = model.l_out(inputs,feat=True)

        predictions = predictions.cpu().data.numpy()

        #final_prediction = np.mean(predictions.cpu().data.numpy(), axis=0)
        # avg_loss = np.mean(loss, axis=0)

        # validation_losses.append(avg_loss)

        # print(predictions.shape)
        # print(id_chunk)

        for i in range(predictions.shape[0]):
            file = open(os.path.join(outputs_path,prelabel+str(id_chunk)+"_"+str(i)+".npy"),"wb")
            np.save(file,predictions[i])
            file.close()

        if n % 1000 == 0:
            print n, 'batches processed'

if train_tta_feat:

    train_it = config().tta_train_data_iterator
    get_preds_targs_tta_feat(train_it)

if train or train_tta:
    if train:
        train_it = config().trainset_valid_data_iterator
        preds, targs, ids = get_preds_targs(train_it)
    elif train_tta:
        train_it = config().tta_train_data_iterator
        preds, targs, ids = get_preds_targs_tta(train_it)
    if dump:
        file = open(prediction_dump,"wb")
        cPickle.dump([preds,targs,ids],file)
        file.close()


if valid_tta_feat:

    valid_it = config().tta_valid_data_iterator
    get_preds_targs_tta_feat(valid_it)


if valid or valid_tta:

    if valid:
        valid_it = config().valid_data_iterator
        preds, targs, ids = get_preds_targs(valid_it)
    elif valid_tta or valid_tta_feat:
        valid_it = config().tta_valid_data_iterator
        preds, targs, ids = get_preds_targs_tta(valid_it)


    if dump:
        file = open(prediction_dump,"wb")
        cPickle.dump([preds,targs,ids],file)
        file.close()

    # weather_targs = []
    # weather_preds = []
    # for t in targs:
    #     weather_targs.append(np.argmax(t[:4]))
    # for p in preds:
    #     weather_preds.append(np.argmax(p[:4]))
    # print weather_preds[:10]
    # print weather_targs[:10]
    # print sklearn.metrics.confusion_matrix(weather_targs,weather_preds)

    print 'Calculating F2 scores'
    threshold = 0.5
    qpreds = preds > threshold
    print app.f2_score(targs[:,:17], qpreds[:,:17])
    print app.f2_score(targs[:,:17], qpreds[:,:17], average=None)
    print 'Calculating F2 scores (argmax for weather class)'
    w_pred = preds[:,:4]
    cw_pred = np.argmax(w_pred,axis=1)
    qw_pred = np.zeros((preds.shape[0],4))
    qw_pred[np.arange(preds.shape[0]),cw_pred] = 1
    qpreds[:,:4] = qw_pred
    print app.f2_score(targs[:,:17], qpreds[:,:17])
    print app.f2_score(targs[:,:17], qpreds[:,:17], average=None)
    print 'Calculating F2 scores only for weather labels'
    print app.f2_score(targs[:,:4], qpreds[:,:4])
    print app.f2_score(targs[:,:4], qpreds[:,:4], average=None)


    print 'loglosses'
    print app.logloss(preds.flatten(), targs.flatten())
    print [app.logloss(preds[:,i], targs[:,i]) for i in range(17)]
    print [sklearn.metrics.log_loss(targs[:,i], preds[:,i], eps=1e-7) for i in range(17)]
    # print 'logloss sklearn'
    # print sklearn.metrics.log_loss(targs, preds)
    # print sklearn.metrics.log_loss(targs.flatten(), preds.flatten(), eps=1e-7)
    print 'skewed loglosses'
    print app.logloss(preds.flatten(), targs.flatten(),skewing_factor=5.)
    print [app.logloss(preds[:,i], targs[:,i],skewing_factor=5.) for i in range(17)]

    tps = [np.sum(qpreds[:,i]*targs[:,i]) for i in range(17)]
    fps = [np.sum(qpreds[:,i]*(1-targs[:,i])) for i in range(17)]
    fns = [np.sum((1-qpreds[:,i])*targs[:,i]) for i in range(17)]

    print 'TP'
    print np.int32(tps)
    print 'FP'
    print np.int32(fps)
    print 'FN'
    print np.int32(fns)

    print 'worst classes'
    print app.get_headers()
    print 4*np.array(fps)+np.array(fns)

if test_tta_feat:

    test_it = config().tta_test_data_iterator
    get_preds_targs_tta_feat(test_it,prelabel='test_')



    test2_it = config().tta_test2_data_iterator
    get_preds_targs_tta_feat(test2_it,prelabel='file_')


if test or test_tta:

    imgid2pred = {}
    imgid2raw = {}
    if test:
        test_it = config().test_data_iterator
        preds, _, ids = get_preds_targs(test_it)
    elif test_tta:
        test_it = config().tta_test_data_iterator
        preds, _, ids = get_preds_targs_tta(test_it)
    for i, p in enumerate(preds):
        imgid2pred['test_'+str(i)] = p > 0.5
        imgid2raw['test_' + str(i)] = p

    if test:
        test2_it = config().test2_data_iterator
        preds, _, ids = get_preds_targs(test2_it)
    elif test_tta:
        test2_it = config().tta_test2_data_iterator
        preds, _, ids = get_preds_targs_tta(test2_it)
    for i, p in enumerate(preds):
        imgid2pred['file_'+str(i)] = p > 0.5
        imgid2raw['file_' + str(i)] = p

    #do not forget argmax for weather labels
    print 'writing submission'
    submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    output_csv_file = submissions_dir + '/%s-%s.csv' % (expid, args.eval)
    submission.write(imgid2pred, output_csv_file)

    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    outputs_path = predictions_dir + '/' + expid
    utils.auto_make_dir(outputs_path)
    output_pickle_file = outputs_path + '/%s-%s.pkl' % (expid, args.eval)
    file = open(output_pickle_file,"wb")
    cPickle.dump(imgid2raw,file)
    file.close()





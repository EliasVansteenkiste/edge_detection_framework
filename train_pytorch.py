import cPickle as pickle
import string
import sys
import time
from itertools import izip

import numpy as np

from datetime import datetime, timedelta


import buffering
import utils
import logger
from configuration import config, set_configuration
import pathfinder
import app
import torch
from torch.autograd import Variable

if len(sys.argv) < 2:
    sys.exit("Usage: CUDA_VISIBLE_DEVICES=<gpu_number> python train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_pytorch', config_name)
expid = utils.generate_expid(config_name)
print
print "Experiment ID: %s" % expid
print

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = metadata_dir + '/%s.pkl' % expid

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)
sys.stderr = sys.stdout

print 'Build model'
model = config().build_model()
model.l_out.cuda() # move to gpu

criterion = config().build_objective()

learning_rate_schedule = config().learning_rate_schedule
learning_rate =(learning_rate_schedule[0])
optimizer = config().build_updates(model.l_out, learning_rate)


chunk_idxs = range(config().max_nchunks)
losses_eval_train = []
losses_eval_valid = []
losses_eval_train2 = []
losses_eval_valid2 = []
start_chunk_idx = 0

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

print
print 'Train model'
chunk_idx = 0
start_time = time.time()
prev_time = start_time

tmp_preds = []
tmp_gts = []

tmp_losses_train = []
tmp_losses_train2 = []
tmp_preds_train = []
tmp_gts_train = []

losses_train_print = []
losses_train_print2 = []
preds_train_print = []
gts_train_print = []
losses_time_print = []

# use buffering.buffered_gen_threaded()
for chunk_idx, (x_chunk_train, y_chunk_train, id_train) in izip(chunk_idxs, buffering.buffered_gen_threaded(
        train_data_iterator.generate(), buffer_size=128)):

    if chunk_idx in learning_rate_schedule:
        lr = learning_rate_schedule[chunk_idx]
        print '  setting learning rate to %.7f' % lr
        print

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    for gt in y_chunk_train:
        tmp_gts.append(gt)
        tmp_gts_train.append(gt)
        gts_train_print.append(gt)

    # make nbatches_chunk iterations
    for b in xrange(config().nbatches_chunk):

        losses_time_print.append(time.time())

        # wrap them in Variable
        inputs, labels = Variable(torch.from_numpy(x_chunk_train).cuda()), \
                         Variable(torch.from_numpy(y_chunk_train).cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.l_out(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pr=outputs.cpu().data.numpy()
        tmp_preds.append(pr)
        tmp_preds_train.append(pr)
        preds_train_print.append(pr)

        tmp_losses_train.append(loss.cpu().data.numpy()[0])

        losses_train_print.append(loss.cpu().data.numpy()[0])

    if (chunk_idx + 1) % 10 == 0:
        print 'Chunk %d/%d %.1fHz' % (chunk_idx + 1, config().max_nchunks,
                                      10. * config().nbatches_chunk * config().batch_size / (
                                      time.time() - losses_time_print[0])),
        print np.mean(losses_train_print)
        print 'score', config().score(gts_train_print, np.vstack(preds_train_print))
        preds_train_print = []
        gts_train_print = []
        losses_train_print = []
        losses_time_print = []
        losses_train_print2 = []
        losses_time_print2 = []

    if ((chunk_idx + 1) % config().validate_every) == 0:
        print
        print 'Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks)
        # calculate mean train loss since the last validation phase
        mean_train_loss = np.mean(tmp_losses_train)

        mean_train_score = np.mean(config().score(tmp_gts_train, np.vstack(tmp_preds_train)))
        print 'Mean train loss: %7f' % mean_train_loss, mean_train_score
        losses_eval_train.append(mean_train_loss)

        tmp_losses_train = []

        tmp_preds_train = []
        tmp_gts_train = []

        # load validation data to GPU

        tmp_losses_valid = []
        tmp_losses_valid2 = []
        tmp_preds_valid = []
        tmp_gts_valid = []
        for i, (x_chunk_valid, y_chunk_valid, ids_batch) in enumerate(
                buffering.buffered_gen_threaded(valid_data_iterator.generate(),
                                                buffer_size=2)):
            inputs, labels = Variable(torch.from_numpy(x_chunk_valid).cuda()), Variable(
                torch.from_numpy(y_chunk_valid).cuda())

            outputs = model.l_out(inputs)
            loss = criterion(outputs, labels)

            pr = outputs.cpu().data.numpy()
            tmp_preds_valid.append(pr)

            tmp_losses_valid.append(loss.cpu().data.numpy()[0])

            for gt in y_chunk_valid:
                tmp_gts_valid.append(gt)


        # calculate validation loss across validation set
        valid_loss = np.mean(tmp_losses_valid)

        valid_score = np.mean(config().score(tmp_gts_valid, np.vstack(tmp_preds_valid)))
        print 'Validation loss: ', valid_loss, valid_score
        losses_eval_valid.append(valid_loss)


        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (config().max_nchunks - chunk_idx + 1.) / (chunk_idx + 1. - start_chunk_idx)
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

    if ((chunk_idx + 1) % config().save_every) == 0:
        print
        print 'Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks)
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration_file': config_name,
                'git_revision_hash': utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': chunk_idx,
                'losses_eval_train': losses_eval_train,
                'losses_eval_valid': losses_eval_valid,
                'param_values': model.l_out.state_dict(),
                'optimizer_values': optimizer.state_dict(),
            }, f, pickle.HIGHEST_PROTOCOL)
            print '  saved to %s' % metadata_path
            print
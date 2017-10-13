import torch
from torch.autograd import Variable
#import cPickle as pickle
import pickle
import string
import sys
import time
import numpy as np


from datetime import datetime, timedelta


import buffering
import utils
import logger
from configuration import config, set_configuration
import pathfinder
import app

print('train.py importing complete')

if len(sys.argv) < 2:
    sys.exit("Usage: CUDA_VISIBLE_DEVICES=<gpu_number> python train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs', config_name)
expid = utils.generate_expid(config_name)

print("\nExperiment ID: %s\n" % expid)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = metadata_dir + '/%s.pkl' % expid
metadata_best_path = metadata_dir + '/%s-best.pkl' % expid
# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)
sys.stderr = sys.stdout

print('Build model')
model = config().build_model()
print(model.l_out)
model.l_out.cuda()    # move to gpu

criterion = config().build_objective()
criterion2 = config().build_objective2()

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

print('\nData')
print('\nn train: %d' % train_data_iterator.nsamples)
print('\n n validation: %d' % valid_data_iterator.nsamples)
print('\n n chunks per epoch', config().nchunks_per_epoch)

print('\nTrain model')
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

best_valid_f2_score = 0
best_threshold = 0.91

# use buffering.buffered_gen_threaded()
for chunk_idx, (x_chunk_train, y_chunk_train, id_train) in zip(chunk_idxs, buffering.buffered_gen_threaded(
        train_data_iterator.generate(), buffer_size=128)):

    if chunk_idx in learning_rate_schedule:
        lr = learning_rate_schedule[chunk_idx]
        print('  setting learning rate to %.7f' % lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # for gt in y_chunk_train:
    #     tmp_gts.append(gt)
    #     tmp_gts_train.append(gt)
    #     gts_train_print.append(gt)

    # make nbatches_chunk iterations
    model.l_out.train()
    for b in range(config().nbatches_chunk):

        losses_time_print.append(time.time())

        # wrap them in Variable
        inputs, labels = Variable(torch.from_numpy(x_chunk_train).cuda()), \
                         Variable(torch.from_numpy(y_chunk_train).cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.l_out(inputs)
        loss = criterion(outputs, labels)
        loss2 = criterion2(outputs, labels)
        loss.backward()
        optimizer.step()

        # pr=outputs.cpu().data.numpy()
        # tmp_preds.append(pr)
        # tmp_preds_train.append(pr)
        # preds_train_print.append(pr)

        loss_out = loss.cpu().data.numpy()[0]
        loss2_out = loss2.cpu().data.numpy()[0]

        tmp_losses_train.append(loss_out)
        tmp_losses_train2.append(loss2_out)
        losses_train_print.append(loss_out)
        losses_train_print2.append(loss2_out)

    if (chunk_idx + 1) % 10 == 0:
        print('Chunk %d/%d %.1fHz' % (chunk_idx + 1, config().max_nchunks,
                                      10. * config().nbatches_chunk * config().batch_size / (
                                      time.time() - losses_time_print[0])),)
        print(np.mean(losses_train_print), np.mean(losses_train_print2))
        # print('score', config().score(np.vstack(preds_train_print), gts_train_print))
        preds_train_print = []
        gts_train_print = []
        losses_train_print = []
        losses_time_print = []
        losses_train_print2 = []
        losses_time_print2 = []

    if ((chunk_idx + 1) % config().validate_every) == 0:
        print('\nChunk %d/%d' % (chunk_idx + 1, config().max_nchunks))
        # calculate mean train loss since the last validation phase
        mean_train_loss = np.mean(tmp_losses_train)
        mean_train_loss2 = np.mean(tmp_losses_train2)
        # mean_train_score = np.mean(config().score(np.vstack(tmp_preds_train), tmp_gts_train))
        mean_train_score = -1
        print('\nMean train loss: %7f' % mean_train_loss, mean_train_loss2, mean_train_score)
        losses_eval_train.append(mean_train_loss)

        tmp_losses_train = []
        tmp_losses_train2 = []
        tmp_preds_train = []
        tmp_gts_train = []

        # load validation data to GPU

        tmp_losses_valid = []
        tmp_losses_valid2 = []

        tmp_preds_valid = []
        tmp_gts_valid = []
        tmp_xs_valid = []

        model.l_out.eval()
        for i, (x_chunk_valid, y_chunk_valid, ids_batch) in enumerate(
                buffering.buffered_gen_threaded(valid_data_iterator.generate(),
                                                buffer_size=2)):
            inputs, labels = Variable(torch.from_numpy(x_chunk_valid).cuda(),volatile=True), Variable(
                torch.from_numpy(y_chunk_valid).cuda(),volatile=True)

            outputs = model.l_out(inputs)
            loss = criterion(outputs, labels)
            loss2 = criterion2(outputs, labels)

            pr = outputs.cpu().data.numpy()
            tmp_preds_valid.append(pr)

            tmp_losses_valid.append(loss.cpu().data.numpy()[0])
            tmp_losses_valid2.append(loss2.cpu().data.numpy()[0])

            for gt in y_chunk_valid:
                tmp_gts_valid.append(gt)
            for xx in x_chunk_valid:
                tmp_xs_valid.append(xx)


        # calculate validation loss across validation set
        valid_loss = np.mean(tmp_losses_valid)
        valid_loss2 = np.mean(tmp_losses_valid2)
        # valid_score = np.mean(config().score(np.vstack(tmp_preds_valid), tmp_gts_valid))
        valid_score = -1
        print('\nValidation loss: ', valid_loss, valid_loss2, valid_score)
        losses_eval_valid.append(valid_loss)
        losses_eval_valid2.append(valid_loss2)


        # do something with the intermediate valid predictions, like saving to image
        config().intermediate_valid_predictions(tmp_xs_valid, tmp_gts_valid, tmp_preds_valid, expid, chunk_idx)


        if valid_score > best_threshold and valid_score > best_valid_f2_score:
            with open(metadata_best_path, 'wb') as f:
                pickle.dump({
                    'configuration_file': config_name,
                    'git_revision_hash': utils.get_git_revision_hash(),
                    'experiment_id': expid,
                    'chunks_since_start': chunk_idx,
                    'losses_eval_train': losses_eval_train,
                    'losses_eval_valid': losses_eval_valid,
                    'param_values': model.l_out.state_dict(),
                    #'optimizer_values': optimizer.state_dict(),
                }, f, pickle.HIGHEST_PROTOCOL)
                print('\n  saved to %s\n' % metadata_best_path)

            best_valid_f2_score = valid_score

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (config().max_nchunks - chunk_idx + 1.) / (chunk_idx + 1. - start_chunk_idx)
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print("  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev))
        print("  estimated %s to go (ETA: %s)\n" % (utils.hms(est_time_left), eta_str))

    if ((chunk_idx + 1) % config().save_every) == 0:

        print('Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks))
        print('Saving metadata, parameters')

        with open(metadata_path, 'wb') as f:
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
            print('  saved to %s\n' % metadata_path)

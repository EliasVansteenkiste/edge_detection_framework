import string
import sys
import lasagne as nn
import numpy as np
import theano
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
from torch.autograd import Variable

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test.py <configuration_name> <test/valid/feat>")

config_name = sys.argv[1]
set_configuration('configs_pytorch', config_name)


valid = sys.argv[2] =='valid'
test = sys.argv[2] == 'test'
feat = sys.argv[2] == 'feat'

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

print("logs")
# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout
print("prediction path")
# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
utils.auto_make_dir(outputs_path)

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

    for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):

        inputs, labels = Variable(torch.from_numpy(x_chunk).cuda()), Variable(
            torch.from_numpy(y_chunk).cuda())

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

    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    print 'Validation loss', np.mean(validation_losses)

    return preds, targs




if valid:
    valid_it = config().valid_data_iterator
    preds, targs = get_preds_targs(valid_it)


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

if test:
    imgid2pred = {}
    test_it = config().test_data_iterator
    preds, _ = get_preds_targs(test_it)
    for i, p in enumerate(preds):
        imgid2pred['test_'+str(i)] = app.apply_argmax_threshold(p)

    test2_it = config().test2_data_iterator
    preds, _ = get_preds_targs(test2_it)
    for i, p in enumerate(preds):
        imgid2pred['file_'+str(i)] = app.apply_argmax_threshold(p)

    #do not forget argmax for weather labels
    print 'writing submission'
    submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    output_csv_file = submissions_dir + '/%s-%s.csv' % (expid, sys.argv[2])
    submission.write(imgid2pred, output_csv_file)






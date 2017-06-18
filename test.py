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

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test.py <configuration_name> <train/test/valid/tta/feat>")

config_name = sys.argv[1]
set_configuration('configs', config_name)


valid = sys.argv[2] =='valid'
test = sys.argv[2] == 'test'
feat = sys.argv[2] == 'feat'
train = sys.argv[2] == 'train'
tta = sys.argv[2] == 'tta'

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
utils.auto_make_dir(outputs_path)

print 'Build model'
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.l_out)
all_params = nn.layers.get_all_params(model.l_out)
num_params = nn.layers.count_params(model.l_out)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

valid_loss = config().build_objective(model, deterministic=True)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared
givens_valid[model.l_target.input_var] = y_shared

# theano functions
if valid or test or train:
    iter_get = theano.function([], [valid_loss, nn.layers.get_output(model.l_out, deterministic=True)],
                                       givens=givens_valid)
elif feat:
    iter_get = theano.function([], [valid_loss, nn.layers.get_output(model.l_feat, deterministic=True)],
                                       givens=givens_valid)
else:
    raise



if test:
    data_iterator = config().test_data_iterator
elif train:
    data_iterator = config().train_data_iterator2
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
        # load chunk to GPU
        # if n == 100:
        #     break
        x_shared.set_value(x_chunk)
        y_shared.set_value(y_chunk)
        loss, predictions = iter_get()
        validation_losses.append(loss)
        targs.append(y_chunk)
        ids.append(id_chunk)
        if feat:
            for idx, img_id in enumerate(id_chunk):
                np.savez(open(outputs_path+'/'+str(img_id)+'.npz', 'w') , features = predictions[idx])

        preds.append(predictions)
        #print id_chunk, targets, loss
        if n%50 ==0:
            print n, 'batches processed'

    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    ids = np.concatenate(ids)
    print 'Validation loss', np.mean(validation_losses)

    return preds, targs, ids


if train:
    train_it = config().train_data_iterator2
    preds, targs, ids = get_preds_targs(train_it)


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

    label_arr = app.get_labels_array()
    for i in range(17):
        fn = (1-qpreds[:,i])*targs[:,i]
        indices = np.int32(np.where(fn==1)[0])
        fn_img_ids = [ids[j] for j in indices]
        for img_id in fn_img_ids:
            print label_arr[img_id,i],
            if label_arr[img_id,i] != 1:
                print 'Warning ', img_id, 'does not have the correct label'
        print 
        print
        print i
        print
        for iid in fn_img_ids:
            print str(iid)+',',
        print
        np.savez(open(outputs_path+'/fn_class_'+str(i)+'.npz', 'w') , idcs = fn_img_ids)

if valid:
    valid_it = config().valid_data_iterator
    preds, targs, ids = get_preds_targs(valid_it)


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
    preds, _, ids = get_preds_targs(test_it)
    for i, p in enumerate(preds):
        imgid2pred['test_'+str(i)] = app.apply_argmax_threshold(p)

    test2_it = config().test2_data_iterator
    preds, _, ids = get_preds_targs(test2_it)
    for i, p in enumerate(preds):
        imgid2pred['file_'+str(i)] = app.apply_argmax_threshold(p)

    print len(imgid2pred), 'predictions'
    #do not forget argmax for weather labels
    print 'writing submission'
    submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    output_csv_file = submissions_dir + '/%s-%s.csv' % (expid, sys.argv[2])
    print output_csv_file
    submission.write(imgid2pred, output_csv_file)






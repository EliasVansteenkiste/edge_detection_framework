import string
import sys
import lasagne as nn
import numpy as np
import theano
import sklearn


import buffering
import pathfinder
import utils
from configuration import config, set_configuration
import logger
import app

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test.py <configuration_name> <'test'/'valid'>")

config_name = sys.argv[1]
set_configuration('configs', config_name)

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
iter_get_predictions = theano.function([], [valid_loss, nn.layers.get_output(model.l_out, deterministic=True)],
                                       givens=givens_valid)


valid = sys.argv[2] =='valid'
test = sys.argv[2] == 'test'
if valid:
    data_iterator = config().valid_data_iterator
elif test:
    data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n', sys.argv[2], ': %d' % data_iterator.nsamples
threshold = 0.5

validation_losses = []
preds = []
targs = []
for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(data_iterator.generate())):
    # load chunk to GPU
    x_shared.set_value(x_chunk)
    y_shared.set_value(y_chunk)
    loss, predictions = iter_get_predictions()
    validation_losses.append(loss)
    targs.append(y_chunk)
    preds.append(predictions)
    #print id_chunk, targets, loss
    if n%50 ==0:
        print n, 'batches processed'

preds = np.concatenate(preds)
targs = np.concatenate(targs)

print preds.shape
print targs.shape


print 'Validation loss', np.mean(validation_losses)
if valid:
    # weather_targs = []
    # weather_preds = []
    # for t in targs:
    #     weather_targs.append(np.argmax(t[:4]))
    # for p in preds:
    #     weather_preds.append(np.argmax(p[:4]))
    # print weather_preds[:10]
    # print weather_targs[:10]
    # print sklearn.metrics.confusion_matrix(weather_targs,weather_preds)

    print 'Calculating F2 scores for each label seperately'
    threshold = 0.5
    qpreds = preds > threshold
    print app.f2_score(targs[:,:17], qpreds[:,:17])
    print app.f2_score(targs[:,:17], qpreds[:,:17], average=None)




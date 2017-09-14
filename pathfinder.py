import json
import utils
import os



with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)

METADATA_PATH = paths["METADATA_PATH"]
utils.check_data_paths(METADATA_PATH)

PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]
utils.check_data_paths(METADATA_PATH)

DATA_PATH = paths["DATA_PATH"]
utils.check_data_paths(DATA_PATH)

# VALIDATION_SPLIT_PATH = paths["VALIDATION_SPLIT_PATH"]
# if not os.path.isfile(VALIDATION_SPLIT_PATH):
#     raise ValueError('no validation file')


import json
import utils
import os



with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)

METADATA_PATH = paths["METADATA_PATH"]
utils.check_data_paths(METADATA_PATH)

DATA_PATH = paths["DATA_PATH"]
utils.check_data_paths(DATA_PATH)

SAMPLE_SUBMISSION_PATH = paths["SAMPLE_SUBMISSION_PATH"]
if not os.path.isfile(SAMPLE_SUBMISSION_PATH):
    raise ValueError('no sample submission file')

# VALIDATION_SPLIT_PATH = paths["VALIDATION_SPLIT_PATH"]
# if not os.path.isfile(VALIDATION_SPLIT_PATH):
#     raise ValueError('no validation file')


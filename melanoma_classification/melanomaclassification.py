import argparse
import logging
import os

import pandas as pd
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

from melanoma_classification.utils.logging_configuration import logging_config
from melanoma_classification.utils.load_config import (load_config,
                                                       set_random_seeds)

from melanoma_classification.isicdownload.loadisicmetadata import load_isic_df
from melanoma_classification.isicdownload.download_metadatafiles import (
    download_all_files)
from melanoma_classification.mclass.loadmclassmetadata import load_mclass_df
from melanoma_classification.ISBI_2016.loadISBI2016metadata import (
    load_isbi2016_test_df)

from melanoma_classification.isicdownload.download_images import download
from melanoma_classification.lib.categories import (standardize_columns,
                                                    random_undersample,
                                                    create_sets,
                                                    filter_metadata)
from melanoma_classification.lib.models.models import (
    resnet50_trainDataGen_flow_df, resnet50_testDataGen_flow_df,
    resnet50_pretrained_simple, resnet50_pretrained_twoLayer)
from melanoma_classification.lib.models.callbacks import (
    modelCheckpoint_callback, CSVLogger_callback, Logger_Callback,
    Val_Acc_Callback, learning_rate)

# configuration
if __name__ == '__main__':

    prs = argparse.ArgumentParser(description='Train a neural net')
    prs.add_argument('--config_file_path',
                     help='path to config file',
                     default='example.config.yaml',
                     required=True)
    args = prs.parse_args()
    config_file_path = args.config_file_path
else:
    this_script_path = __file__
    this_script_dir = os.path.abspath(os.path.split(this_script_path)[0])
    config_file_path = os.path.realpath(
        os.path.join(this_script_dir, '../example.config.yaml'))

# logging
logger = logging.getLogger()
logger = logging_config(logger)

# configuration
config_dict = load_config(config_file_path)

# join the metadata paths.
config_dict['download_images']['MClass_metadata'] = os.path.join(
    os.path.abspath(config_dict['metadata_path']),
    config_dict['download_images']['MClass_metadata_filename'])

config_dict['download_images']['isbi2016_test_metadata'] = os.path.join(
    os.path.abspath(config_dict['metadata_path']),
    config_dict['download_images']['isbi2016_test_metadata_filename'])

config_dict['download_images']['isic_images_metadata_path'] = os.path.join(
    os.path.abspath(config_dict['metadata_path']),
    config_dict['download_images']['isic_images_metadata_filename'])

set_random_seeds(config_dict, logger)

logger.info('loading modules, parsing args => Successful')

# ensure the needed directories exist.
# directories needed.
needed = {
    config_dict['download_images']['images_base_path'],
    config_dict['log_path'],
    config_dict['models_path'],
}  # a set
# create directories.
[os.makedirs(os.path.abspath(d), exist_ok=True) for d in needed]
logger.info('creating directories >> Successful.')

# download metadata files
download_all_files(os.path.abspath(config_dict['metadata_path']))

# download the images
images_base_path = config_dict['download_images']['images_base_path']

isic_metadata_path = (
    config_dict['download_images']['isic_images_metadata_path'])
mclass_metadata_path = (config_dict['download_images']['MClass_metadata'])
isbi2016test_metadata_path = (
    config_dict['download_images']['isbi2016_test_metadata'])

isic_metadata_df = load_isic_df(isic_metadata_path)
mclass_metadata_df = load_mclass_df(mclass_metadata_path)
isbi2016test_metadata_df = load_isbi2016_test_df(isbi2016test_metadata_path)
filtered_isic_metadata_df = filter_metadata(isic_metadata_df)

# union
# of: filtered, mclass and isbi2016
# to download:
# the pipe symbol "|" carries out the union.
download_df = isic_metadata_df.loc[(filtered_isic_metadata_df.index
                                    | mclass_metadata_df.index
                                    | isbi2016test_metadata_df.index)]

df = download(download_df, images_base_path)

# standardize image DataFrames, i.e. select only the relevant columns and
# rename them to 'id' and 'category'.
std_df = standardize_columns(df, config_dict['id_column'],
                             config_dict['category_column'])
# in the ISBI 2016 test set there are two images without clear
# category:
# - 'ISIC_0009959' with category 'indeterminate'
# - 'ISIC_0010454' with category 'indeterminate/malignant'
# in the official ground truth both are counted as malignant
std_df[std_df.isin({'category': ['indeterminate',
                                 'indeterminate/malignant']})] = 'malignant'
# 'ISIC_0011319' has no category. in the official ground truth it is treated as
# benign
std_df.loc[std_df['id'] == 'ISIC_0011319.jpg', 'category'] = 'benign'

# select images which are in filtered but not in mclass or isic2016test
# a set difference operation.
filtered_std_df = std_df.loc[filtered_isic_metadata_df.index.difference(
    mclass_metadata_df.index.union(isbi2016test_metadata_df.index))]

# isic 2016 test std
isic2016test_std_df = std_df.loc[isbi2016test_metadata_df.index]
# mclass std
mclass_std_df = std_df.loc[mclass_metadata_df.index]

# random undersample, such that there the same amount of images in each
# category
rus_filtered_std_df = random_undersample(filtered_std_df)

# create training, validation and test set
training_set, validation_set, testing_set = create_sets(
    rus_filtered_std_df, config_dict['validation_size'],
    config_dict['testing_size'])

# create DataGenerators
training_gen = resnet50_trainDataGen_flow_df(
    training_set,
    images_base_path,
    batch_size=config_dict['batch_size'],
    class_mode='binary')
validation_gen = resnet50_testDataGen_flow_df(
    validation_set,
    images_base_path,
    batch_size=config_dict['batch_size'],
    class_mode='binary')

# adapt steps to workflow_testing flag
if config_dict['workflow_testing']:
    steps_per_epoch = 3
    validation_steps = 3
else:
    steps_per_epoch = len(training_gen)
    validation_steps = len(validation_gen)

# models
simple_model = resnet50_pretrained_simple()
twoLayer_model = resnet50_pretrained_twoLayer()


def train(model):
    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-03, momentum=0.9),
                  metrics=['acc'])

    num_non_trainable_epochs = 3

    # train the classification layer
    history = model.fit_generator(
        training_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=num_non_trainable_epochs,
        validation_data=validation_gen,
        validation_steps=validation_steps,
        verbose=1,
        # initial_epoch=epoch,
        callbacks=[
            modelCheckpoint_callback(config_dict['models_path'],
                                     config_dict['model_name']),
            CSVLogger_callback(config_dict['log_path'],
                               config_dict['log_name']),
            LearningRateScheduler(lambda epoch, lr: 10**(-(epoch + 3))),
            Logger_Callback(os.path.join(config_dict['log_path'], 'mylog.csv'),
                            validation_gen=validation_gen),
        ])

    # unfreeze all layers
    for layer in model.layers:
        layer.trainable = True
    #
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-5, momentum=0.9),
                  metrics=['acc'])

    # train the whole network
    history = model.fit_generator(
        training_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=100 + num_non_trainable_epochs,
        validation_data=validation_gen,
        validation_steps=validation_steps,
        verbose=1,
        initial_epoch=num_non_trainable_epochs,
        callbacks=[
            modelCheckpoint_callback(config_dict['models_path'],
                                     config_dict['model_name']),
            CSVLogger_callback(config_dict['log_path'],
                               config_dict['log_name']),
            LearningRateScheduler(
                learning_rate(start_epoch=num_non_trainable_epochs,
                              start_learning_rate=1e-5)),
            Logger_Callback(os.path.join(config_dict['log_path'], 'mylog.csv'),
                            validation_gen=validation_gen,
                            some_layers_frozen=False),
        ])

    return model

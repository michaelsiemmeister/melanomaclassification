import os
import datetime

import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import (Callback, ModelCheckpoint, CSVLogger)
from sklearn.metrics import accuracy_score


def modelCheckpoint_callback(dir_path, name):
    '''
    I:
      dir_path, where to safe the model .. str
      name, ... model name ... str, e.g. 'model1'

    Saves the model

    '''
    filepath = os.path.join(
        os.path.abspath(dir_path),
        name) + '_epoch_{epoch:02d}_valloss_{val_loss:.2f}' + '.hdf5'

    return ModelCheckpoint(filepath, )


def CSVLogger_callback(dir_path, name, append=True):
    filepath = os.path.join(os.path.abspath(dir_path), name) + '.csv'
    return CSVLogger(filepath, append=append)


def custom_validation_accuracy(validation_gen, model):
    ground_truth = np.array(validation_gen.classes)
    # print('\n', ground_truth, '\n')
    predictions = model.predict_generator(validation_gen)
    # print('\n', predictions, '\n')
    predictions = np.ravel(predictions)  # 1-D
    accuracy = accuracy_score(ground_truth, np.around(predictions))
    # print('validation accuracy = {}'.format(accuracy))
    return accuracy


class Logger_Callback(Callback):
    def __init__(self, filepath, some_layers_frozen=True, validation_gen=None):
        super(Logger_Callback, self).__init__()
        self.filepath = filepath
        self.some_layers_frozen = some_layers_frozen
        self.validation_gen = validation_gen

    def on_epoch_end(self, epoch, logs=None):
        # read the pandas DataFrame if it exists:
        if os.path.exists(self.filepath):
            df = pd.read_csv(self.filepath)
        else:
            df = pd.DataFrame()

        this_dict = {k: [v] for (k, v) in logs.items()}
        this_dict['epoch'] = [epoch]
        this_dict['frozen_layers'] = [self.some_layers_frozen]
        this_dict['custom_val_acc'] = [
            custom_validation_accuracy(self.validation_gen, self.model)
        ]
        this_dict['current_time'] = datetime.datetime.isoformat(
            datetime.datetime.utcnow())

        this_epoch_df = pd.DataFrame(this_dict)
        # print('\n')
        # print(df)
        # print(this_epoch_df)

        concated = pd.concat([df, this_epoch_df])
        # print(concated)
        concated.to_csv(self.filepath, index=False)


class Val_Acc_Callback(Callback):
    def __init__(self, validation_gen):
        super(Val_Acc_Callback, self).__init__()
        self.validation_gen = validation_gen

    def on_epoch_end(self, epoch, logs=None):
        # just 1 D numpy array
        ground_truth = np.array(self.validation_gen.classes)
        # print('\n', ground_truth, '\n')
        predictions = self.model.predict_generator(self.validation_gen)
        # print('\n', predictions, '\n')
        predictions = np.ravel(predictions)  # 1-D
        accuracy = accuracy_score(ground_truth, np.around(predictions))
        # print('validation accuracy = {}'.format(accuracy))


def learning_rate(start_epoch=0,
                  one_tenth_after_no_epochs=3,
                  start_learning_rate=1e-5):
    '''
    Returns a function which takes an integer - epoch - as input.
    Output: a float, the new learning rate.
    '''
    def lr(epoch):
        return start_learning_rate * 10**(-(epoch - start_epoch) /
                                          one_tenth_after_no_epochs)

    return lr
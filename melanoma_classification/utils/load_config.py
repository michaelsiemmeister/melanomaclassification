import copy
import os
import random

import numpy as np
import tensorflow

import yaml


def load_config(path_str):
    '''
    I:
        path_str    str     absolute or relative path for config yaml file
    O:
                    dict    dict of dicts or lists containing the configuration
    '''
    with open(os.path.abspath(path_str)) as yamlfile:
        config = yaml.safe_load(yamlfile)
        return config


def set_random_seeds(config_dict, logger):
    logger.debug('setting random seeds')
    random.seed(a=config_dict['random_seeds']['built_in'])
    np.random.seed(seed=config_dict['random_seeds']['numpy'])
    tensorflow.random.set_seed(config_dict['random_seeds']['tensorflow'])
    logger.debug('set random seeds - DONE')

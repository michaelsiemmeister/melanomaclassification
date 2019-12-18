# -*- coding: utf-8 -*-

'''

This module has the goals:

- BASE_SET -> TRAIN SET, VAL SET, TEST SET


In this module the concepts of categories and stages is used.

An image can belong to a category, e.g. 'benign' or 'malignant'.

An image can be used in a stage like 'training' or 'validation' or 'testing'.

'''
import logging
import copy
import random
import math
import itertools

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# configure logging for this module
module_logger = logging.getLogger(__name__)


# set random state
random.seed(a=1)
initial_random_state = random.getstate()


def combine_dataframes(metadata_df, download_df):
    '''
    Specific for my dataset.
    I:
      2 dataframes

    Combines them based on '_id' and 'name' column

    Returns a combined pandas DataFrame
    '''
    return pd.merge(metadata_df, download_df, how='inner', on=['_id', 'name'])


def filter_metadata(metadata_df):
    '''
    My specific filter function.
    Filters the metadata_df.
    Very specific to this project.

    I:
        metadata_df     pandas DataFrame    from ISIC_images_metadata.csv

    O:
        filtered_metadata_df    pandas DataFrame    filtered.
    '''
    filter_df = (
        # only images with known dignity, i.e. benign or malignant
        (metadata_df['meta_clinical_benign_malignant'].isin(
            ['benign', 'malignant']))
        # only melanocytic images
        & (metadata_df['meta_clinical_melanocytic'])
        # only images confirmed by histology
        & (metadata_df['meta_clinical_diagnosis_confirm_type'] == (
            'histopathology'))
        # only dermoscopic images
        & (metadata_df['meta_acquisition_image_type'] == 'dermoscopic'))

    filtered_metadata_df = metadata_df[filter_df]
    return filtered_metadata_df


def standardize_columns(df, id_column, category_column):
    '''
    - only selects the id_column and category_column.
    - renames
        id_column to 'id'
        category_column to 'category'
    I:
        df ... pandas DataFrame object
        id_column ... str
        category_column ... str

    O:
        pandas DataFrame object
    '''
    select_columns_df = df.loc[:, [id_column, category_column]]
    output_df = select_columns_df.rename(
        columns={
            id_column: 'id',
            category_column: 'category'})
    # remove the '.jpg' part
    output_df['index_col'] = output_df['id'].apply(lambda x: x[:-4])
    output_df.set_index('index_col', inplace=True)
    return output_df


def _duplicates(*dfs):
    '''
    check if there are elements in more than 1 dataframe
    I:
        dfs... standardized (column 'id') pandas DataFrames
    O:
        bool..True ( There are  duplicates) or False
    '''
    sets = [set(df['id'])
            for df in dfs]
    intersect = set.intersection(*sets)
    return bool(intersect)


def create_sets(standardized_df, validation_size, test_size):
    '''
    I:
        standardized_df pandas DataFrame, with columns: 'id', 'category'.
        validation_size, test_size:
            float - percentage, or int - absolute count
    O:
        Returns a tuple of 3 pandas DataFrames: training, validation, testing

    Partition the data into 3 sets.
    First split off testing_percentage, then split off validation_percentage
    from the rest.
    '''
    process_further, testing = train_test_split(
        standardized_df,
        test_size=test_size,
        random_state=0,
        stratify=standardized_df['category']
    )
    training, validation = train_test_split(
        process_further,
        test_size=validation_size,
        random_state=0,
        stratify=process_further['category']
    )
    [df.reset_index(inplace=True, drop=True)
     for df in [training, validation, testing]]

    if _duplicates(training, validation, testing):
        raise ValueError('There are duplicates')

    return (training, validation, testing)


def random_undersample(standardized_df):
    '''
    I:
        standardized_df - pandas DataFrame -> columns: ['id', 'category']
    O:
        standardized_df with random_undersampling
    '''
    random_undersampler = RandomUnderSampler(random_state=1)
    x, y = random_undersampler.fit_resample(
        standardized_df.loc[:, ['id']], standardized_df['category'])
    return pd.DataFrame({'id': np.ravel(x), 'category': y})

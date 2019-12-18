

import os

import pandas as pd


def load_isic_df(metadata_path):
    '''
    I:
        metadata_path ... str, relative or absolute path to metadata csv file.

    return pandas DataFrame
    - set the 'name' column as index with name 'id'
    '''
    metadata_path = os.path.abspath(metadata_path)
    df = pd.read_csv(metadata_path, low_memory=False)
    df.set_index('name', drop=False, inplace=True)
    df.index.rename('id', inplace=True)
    return df

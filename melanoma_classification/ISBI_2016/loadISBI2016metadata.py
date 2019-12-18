import os

import pandas as pd


def load_isbi2016_test_df(metadata_path):
    '''
    I:
        metadata_path ... str, relative or absolute path to metadata csv file.

    return pandas DataFrame
    - set the 'name' column as index with name 'id'
    '''
    metadata_path = os.path.abspath(metadata_path)
    df = pd.read_csv(metadata_path,
                     low_memory=False,
                     header=None,
                     names=['id', 'category'])
    df.set_index('id', drop=False, inplace=True)
    return df

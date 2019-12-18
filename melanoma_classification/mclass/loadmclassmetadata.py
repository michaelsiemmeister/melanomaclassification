
import os

import pandas as pd


def load_mclass_df(metadata_path):
    '''
    I:
        metadata_path ... str, relative or absolute path to metadata csv file.

    return pandas DataFrame
      - zeropad the 'File_name' column with 7 zeros and
        convert it to str and prefix it with ISIC_.
      - rename the 'File_name' column to 'id'.
      - set the 'id' column as index.
    '''
    metadata_path = os.path.abspath(metadata_path)
    df = pd.read_csv(metadata_path,
                     converters={
                         'File_name': lambda x: 'ISIC_' + str(x).zfill(7)})
    df.rename(columns={'File_name': 'id'}, inplace=True)
    df.set_index('id', drop=False, inplace=True)
    return df

import math

import pandas as pd
from sklearn.model_selection import train_test_split

import melanoma_classification.lib.categories as categories


def create_test_df(length=30, c1_percentage=0.5):
    '''
    creates a test_df
    I:
        kwarg:
            length.... integer.
            c1_percentage .. float between 0.0 an 1.0
    O:
        pandas DataFrame
    '''
    num_1 = math.floor(length * c1_percentage)
    num_2 = length - num_1
    df = pd.DataFrame(
        {
            'id_column_name': list(range(length)),
            'category_column_name': num_1 * [1] + num_2 * [2],
            'dummy_column': length
        })
    return df


def test_create_test_df():
    df = create_test_df()
    assert(isinstance(df, pd.DataFrame))
    print(df)


def test_standardize_columns():
    df = create_test_df(length=2)
    standardized_df = categories.standardize_columns(
        df,
        'id_column_name',
        'category_column_name'
    )
    assert(list(standardized_df.columns) == ['id', 'category'])
    print(standardized_df)


def test_StratifiedShuffleSplit():
    df = create_test_df()
    standardized_df = categories.standardize_columns(
        df,
        'id_column_name',
        'category_column_name'
    )
    splits = train_test_split(
        standardized_df,
        train_size=0.8,
        random_state=0,
        stratify=standardized_df['category'])

    assert(list(splits[1].groupby('category').size()) == [3, 3])
    assert(list(splits[0].groupby('category').size()) == [12, 12])

    [print(split, split.groupby('category').size()) for split in splits]


def test_create_sets():
    df = create_test_df(length=100)
    standardized_df = categories.standardize_columns(
        df,
        'id_column_name',
        'category_column_name'
    )
    training, validation, testing = categories.create_sets(
        standardized_df, 10, 10)
    print(training)
    print(validation)
    print(testing)
    assert(len(validation) == 10 and len(testing) == 10)


def test_random_undersample():
    df = create_test_df(length=30, c1_percentage=0.8)
    standardized_df = categories.standardize_columns(
        df,
        'id_column_name',
        'category_column_name'
    )
    rus_df = categories.random_undersample(standardized_df)
    print(rus_df)
    assert(
        list(
            rus_df['id'] == [
                13,
                18,
                3,
                14,
                20,
                17,
                24,
                25,
                26,
                27,
                28,
                29]))


def test_duplicates():
    df1 = pd.DataFrame({'id': [1, 2, 3, 4]})
    df2 = pd.DataFrame({'id': [2, 3]})
    df3 = pd.DataFrame({'id': [5, 6]})

    assert(categories._duplicates(df1, df2))
    assert(not categories._duplicates(df1, df3))

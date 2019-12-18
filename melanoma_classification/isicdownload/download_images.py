# -*- coding: utf-8 -*-

import asyncio
import nest_asyncio
nest_asyncio.apply()
import functools
import logging
import os
import random
import sys
import time
from operator import xor

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
import requests
from aiohttp import ClientSession

import melanoma_classification.lib.categories as categories

# configure logging for this module
module_logger = logging.getLogger(__name__)


def batch_list(input_list, num_elements_per_chunk):
    '''
    '''
    for position in range(0, len(input_list), num_elements_per_chunk):
        # yield the batch
        yield input_list[position:position + num_elements_per_chunk]


async def fetch_image(url, session='None', logger=module_logger):
    try:
        resp = await session.request(method='GET', url=url)
        resp.raise_for_status()
        logger.debug("Got response {} for URL {}".format(resp.status, url))
        img = await resp.read()
        return img
    except Exception as e:
        logger.warning(
            f'downloading: \n\n     {url}\n\nfailed.\n\nError message:\n',
            exc_info=True)
        raise


async def write_one_img_to_file(row,
                                session='None',
                                dst=None,
                                logger=module_logger):
    '''
    write image to file
    '''

    logger.debug('called write_one_img_to_file')
    if dst is None:
        # just raise any error for now.
        raise ValueError

    dictionary = row.to_dict()
    dictionary['already_exists'] = None
    dictionary['successful_download'] = None
    dictionary['successful_writeFile'] = None
    dictionary['file_size'] = None
    dictionary['download_name'] = dictionary['name'] + '.jpg'

    image_path = os.path.join(dst, dictionary['download_name'])
    url = 'https://isic-archive.com/api/v1/image/{}/download'.format(
        dictionary['_id'])

    if os.path.isfile(image_path):
        if os.path.getsize(image_path) < 10:
            os.remove(image_path)
        else:
            dictionary['already_exists'] = True
    else:
        dictionary['already_exists'] = False
        try:
            img = await fetch_image(url, session)
            logger.debug('fetched an image')
            dictionary['successful_download'] = True
        except Exception as e:
            dictionary['successful_download'] = False

    if dictionary['already_exists']:
        pass
    elif dictionary['successful_download']:
        try:
            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(img)
            dictionary['successful_writeFile'] = True
        except Exception as e:
            dictionary['successful_writeFile'] = False
            if os.path.isfile(image_path):
                os.remove(image_path)

    # get the file size
    if dictionary['already_exists'] or dictionary['successful_writeFile']:
        dictionary['file_size'] = os.path.getsize(image_path)

    return dictionary


async def bulk_download_and_write(dst,
                                  df,
                                  logger=module_logger,
                                  max_num_connections=5):
    '''
    Download and write concurrently to file multiple images.
    '''
    logger.debug('called bulk_download_and_write')

    limited_connector = aiohttp.TCPConnector(limit=max_num_connections)
    async with ClientSession(connector=limited_connector) as session:
        tasks = []
        for index, row in df.iterrows():
            tasks.append(write_one_img_to_file(row, session=session, dst=dst))

        task_batch_list = list(batch_list(tasks, max_num_connections))

        counter = 0
        results = []
        for batch in task_batch_list:
            batch_result = await asyncio.gather(*batch)
            results.extend(batch_result)
            counter += max_num_connections
            logger.debug(f'processed {counter} images')

    logger.debug(f'processed all {counter} images.')
    return results


def _download(metadata_df,
              IMAGES_BASE_PATH,
              logger=module_logger,
              all_images=True,
              num_images=None):
    '''
    Downloads the images and returns a pandas DataFrame.

    I:
        metadata_df     pandas DataFrame    must have '_id' and 'name' column
        IMAGES_BASE_PATH    str             an absolute path.


    returns:
        return_df... a pandas DataFrame object with the following columns
            _id
            name
            already_exists
            successful_download
            successful_writeFile
            file_size
            success
    '''

    logger.info('downloading images...')
    start = time.time()

    # input checking
    if not xor(bool(all_images), bool(num_images)):
        raise ValueError(
            'specifiy the number of images or download all images')

    if num_images:
        df_download = metadata_df.iloc[0:num_images]
    else:
        df_download = metadata_df

    logger.info(f'There are {len(df_download.index)} images to be downloaded')

    # asyncio.run returns the return value of the given function.
    # in our case, results is a dict.
    results = asyncio.run(
        bulk_download_and_write(IMAGES_BASE_PATH, df_download[['_id',
                                                               'name']]))

    # logger.info(f'{list(iter(results[0]))}')

    # create a DataFrame
    result_df = pd.DataFrame(
        results,
        columns=list(iter(results[0]))
        # columns=['_id', 'name', 'already_exists',
        #          'successful_download', 'successful_writeFile', 'file_size',
        #          'download_name']
    )

    # set a new column in the DataFrame, which inidicates whether the image is
    # downloaded correctly
    result_df['success'] = result_df.apply(
        lambda x: x['file_size'] > 100 and
        (x['successful_writeFile'] or x['already_exists']),
        axis=1)

    # check if processing all the images was successful.
    processing_all_images_successful = result_df['success'].all()

    if processing_all_images_successful:
        logger.info('processed all images successfully')
    else:
        logger.warn('did not process all images successfully')
        raise Exception('did not progress all images successfully')

    logger.debug('\n' + '\n' + f'{result_df}')
    logger.debug(f'elapsed time: {time.time() - start} seconds')

    return result_df


def _download_postprocessing(filtered_metadata_df,
                             download_df,
                             logger=module_logger):
    # combine the dataframes
    combined_df = categories.combine_dataframes(filtered_metadata_df,
                                                download_df)
    logger.debug('combined dataframe: \n' + '\n' + f'{combined_df}')

    # check for duplicates
    bool_duplicates = combined_df['name'].duplicated().all()
    logger.info('there are duplicates: {}'.format(bool_duplicates))
    if bool_duplicates:
        raise ValueError('There are duplicates in the dataframe')

    return combined_df


def download(metadata_df,
             save_directory_path,
             filter_fn=categories.filter_metadata):
    '''
    Highly specific function for ISIC data and this project.
    I:
        metadata_df   pandas dataframe
        save_directory_path  str     path to directory - where to store the
                                     images; relative or absolute

    O:
        pandas Dataframe with download status and metadata

    '''
    #  abspath
    save_directory_path = os.path.abspath(save_directory_path)

    download_df = _download(metadata_df, save_directory_path)
    combined_df = _download_postprocessing(metadata_df, download_df)
    return combined_df

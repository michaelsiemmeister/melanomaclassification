import os
from io import StringIO, BytesIO
import logging
import csv

import requests
import pandas as pd

log = logging.getLogger(__name__)

isic_link_test = (
    'https://challenge.kitware.com/api/v1/item/56faf116cad3a5465d878cfb/download'
)
isic_link_training = (
    'https://challenge.kitware.com/api/v1/file/568e86dacad3a5219e3f45a2/download'
)

mclass_resultsdermoscopic_link = (
    'https://skinclass.de/MClass/ResultsDermoscopic.xlsx')
mclass_dermoscopicNamesource_link = (
    'https://skinclass.de/MClass/DermoscopicNameSource.xlsx')
resultsDermoscopic_name = 'MClass_ResultsDermoscopic.csv'
NameSource_name = 'MClass_DermoscopicNameSource.csv'


def download_isic_ground_truth(link, path):
    try:
        r = requests.get(link)
        if r.status_code == 200:
            with open(path, 'bw') as f:
                f.write(r.content)
        else:
            log.critical('link {} could not be downloaded'.format(link))
    except:
        log.critical('link {} could not be downloaded'.format(link))
    return None


def download_isic_ground_truth_files(path):
    for fn, l in zip(('ISBI2016_ISIC_Part3_Test_GroundTruth.csv',
                      'ISBI2016_ISIC_Part3_Training_GroundTruth.csv'),
                     (isic_link_test, isic_link_training)):

        file_path = os.path.abspath(os.path.join(path, fn))
        # print(file_path)
        # print(l)
        download_isic_ground_truth(l, file_path)


def download_resultsDermoscopic(path):
    try:
        r = requests.get(mclass_resultsdermoscopic_link)
        if r.status_code == 200:
            bIO = BytesIO(r.content)
            df = pd.read_excel(bIO, header=1)
            filepath = os.path.join(os.path.abspath(path),
                                    resultsDermoscopic_name)
            df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
        else:
            raise Exception('wrong statuscode')
    except:
        log.critical('link {} could not be downloaded'.format(
            mclass_dermoscopicNamesource_link))
    return None


def download_DermoscopicNameSource(path):
    try:
        r = requests.get(mclass_dermoscopicNamesource_link)
        if r.status_code == 200:
            bIO = BytesIO(r.content)
            df = pd.read_excel(bIO, header=0, usecols=[0, 1])
            df.rename(columns={'file name': 'File_name'}, inplace=True)

            filepath = os.path.join(os.path.abspath(path), NameSource_name)
            df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
        else:
            raise Exception('wrong statuscode')
    except:
        log.critical('link {} could not be downloaded'.format(
            mclass_dermoscopicNamesource_link))
    return None


def check_if_metadata_ready(path):
    p = os.path.abspath(path)
    files_set = set(os.listdir(p))
    needed_set = {
        'ISBI2016_ISIC_Part3_Test_GroundTruth.csv',
        'ISBI2016_ISIC_Part3_Training_GroundTruth.csv',
        resultsDermoscopic_name, NameSource_name
    }
    return needed_set <= files_set


def download_all_files(path):
    path = os.path.abspath(path)
    if not check_if_metadata_ready(path):
        download_isic_ground_truth_files(path)
        download_resultsDermoscopic(path)
        download_DermoscopicNameSource(path)
    return True

import pandas as pd

import melanoma_classification.isicdownload.download_images as download_images


def test_download():
    metadata_path = 'ISIC_images_metadata.csv'
    save_dir_path = '../Images'
    df = download_images.download(metadata_path, save_dir_path)
    assert(isinstance(df, pd.DataFrame))

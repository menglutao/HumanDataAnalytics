import gzip
import logging
import os
import shutil
import tarfile
import zipfile



# unnecessary now but maybe useful later
def make_dir(file_path):
    """
    package tar.gz file
    :param file_path: target file path
    :param process_id: current start edge id
    :return: bool
    """
    try:
        os.makedirs(file_path)
        return True
    except Exception as e:
        logging.exception(e)
        return False
    
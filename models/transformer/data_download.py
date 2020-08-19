# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import random
import tarfile
import urllib.request

import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.INFO)


_TRAIN_DATA = {
    'url': 'http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
    'inputs': 'news-commentary-v12.de-en.en',
    'targets': 'news-commentary-v12.de-en.de'
}

_EVAL_DATA = {
    'url': 'http://data.statmt.org/wmt17/translation-task/dev.tgz',
    'inputs': 'newstest2013.en',
    'targets': 'newstest2013.de'
}

_TEST_DATA = {
    'url': 'https://storage.googleapis.com/tf-perf-public/official_transformer/test_data/newstest2014.tgz',
    'inputs': 'newstest2014.en',
    'targets': 'newstest2014.de'
}


def find_file(root_path, filename, max_depth=6):
    for root, dirs, files in os.walk(root_path):
        if filename in files:
            return os.path.join(root_path, filename)

        depth = root[len(root_path) + 1:].count(os.sep) + 1
        if depth > max_depth:
            # 这里不能直接 break
            # 因为 os.walk 是深度优先搜索
            del dirs

    return None


def download_report_hook(block_num, block_size, total_size):
    p = int(block_num * block_size * 100 / total_size)
    print('\r%d%%' % p + ' completed', end='\r')


def download_data_from_url(save_path, data_url):
    filename = os.path.join(save_path, data_url.split('/')[-1])
    logging.info('download %s from %s' % (filename, data_url))

    incomplete_download_file = filename + '.incomplete'
    incomplete_download_file, _ = urllib.request.urlretrieve(
        url=data_url,
        filename=incomplete_download_file,
        reporthook=download_report_hook
    )
    os.rename(incomplete_download_file, filename)
    return filename


def download_and_extract_file(
        save_path,
        data_url,
        inputs_filename,
        targets_filename
):
    """
        1. 判断文件是否已经存在，若存在，则直接返回文件路径
        2. 判断压缩文件是否存在，若存在，转到 4
        3. 若不存在，则从 data url 下载压缩文件
        4. 解压缩
        5. 返回所需文件
    """

    # 1. 判断文件是否已经存在
    inputs_file = find_file(save_path, inputs_filename)
    targets_file = find_file(save_path, targets_filename)
    if inputs_file and targets_file:
        logging.info('inputs_file and targets_file already exist, return directly')
        return inputs_file, targets_file

    # 2. 判断压缩文件是否存在
    compressed_filename = data_url.split('/')[-1]
    compressed_file = find_file(save_path, compressed_filename)

    # 3. 若不存在，下载压缩文件
    if not compressed_file:
        compressed_file = download_data_from_url(save_path, data_url)

    # 4. 解压缩到 save path
    logging.info('extract %s to %s' % (compressed_file, save_path))
    with tarfile.open(compressed_file) as f:
        f.extractall(save_path)

    # 5. 找到所需文件并返回
    inputs_file = find_file(save_path, inputs_filename)
    targets_file = find_file(save_path, targets_filename)
    if inputs_file and targets_file:
        return inputs_file, targets_file

    raise OSError(
        'not found inputs_file and targets_file',
        'maybe failed in downloading or extracting'
    )


def make_dir(path):
    if not tf.io.gfile.exists(path):
        logging.info('create save path %s' % path)
        tf.io.gfile.makedirs(path)


def main(save_path):
    make_dir(save_path)
    train_files = download_and_extract_file(
        save_path,
        _TRAIN_DATA['url'],
        _TRAIN_DATA['inputs'],
        _TRAIN_DATA['targets']
    )
    eval_files = download_and_extract_file(
        save_path,
        _EVAL_DATA['url'],
        _EVAL_DATA['inputs'],
        _EVAL_DATA['targets']
    )
    test_files = download_and_extract_file(
        save_path,
        _TEST_DATA['url'],
        _TEST_DATA['inputs'],
        _TEST_DATA['targets']
    )
    return train_files, eval_files, test_files


if __name__ == '__main__':
    main('D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende')

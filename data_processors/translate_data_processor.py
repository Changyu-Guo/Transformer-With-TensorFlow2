# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tarfile
import urllib.request
import tensorflow as tf
from absl import logging

_PREFIX = ''
_TRAIN_TAG = ''
_EVAL_TAG = ''

_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1

_TRAIN_DATA_SOURCES = [
    {
        'url': 'http://data.statmt.org/wmt17/translation-task/'
               'training-parallel-nc-v12.tgz',
        'input': 'news-commentary-v12.de-en.en',
        'targets': 'news-commentary-v12.de-en.de'
    }
]


def find_file(path, filename, max_depth=5):
    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)

        depth = root[len(path) + 1:].count(os.sep)

        # 停止递归
        if depth > max_depth:
            del dirs[:]

    return None


def get_raw_files(raw_dir, data_source):
    raw_files = {
        'inputs': [],
        'targets': []
    }

    for d in data_source:
        input_file, target_file = download_and_extract(
            raw_dir, d['url'], d['input'], d['target']
        )

        raw_files['inputs'].append(input_file)
        raw_files['targets'].append(target_file)

    return raw_files


def download_report_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print('\r%d%%' % percent + ' completed', end='\r')


def download_from_url(path, url):
    """
    :param path: 文件保存路径
    :param url: 文件下载路径
    :return:
    """

    # 确认压缩包是否已经下载
    filename = url.split('/')[-1]
    found_file = find_file(path, filename, max_depth=0)
    if found_file is None:
        filename = os.path.join(path, filename)
        logging.info('Downloading from %s to %s.' % (url, filename))
        in_progress_filename = filename + '.incomplete'
        in_progress_filename, _ = urllib.request.urlretrieve(
            url, in_progress_filename, reporthook=download_report_hook
        )
        print()
        tf.io.gfile.rename(in_progress_filename, filename)
        return filename
    else:
        logging.info('Already downloaded: %s (at %s).' % (url, found_file))
        return found_file


def download_and_extract(path, url, input_filename, target_filename):
    """
    :param path: 文件下载路径（文件夹）
    :param url: 获取压缩后的双语翻译语料的 URL
    :param input_filename: 解压双语语料得到的输入文件
    :param target_filename: 解压双语语料得到的输出文件
    :return: 输入语料文件名和输出语料文件名
    """

    # 确认是否已经下载
    input_file = find_file(path, input_filename)
    target_file = find_file(path, target_filename)
    if input_file and target_file:
        logging.info('Already downloaded and extracted %s.' % url)
        return input_file, target_file

    # 语料未下载则进行下载
    compressed_file = download_from_url(path, url)

    # 解压缩
    logging.info('Extracting %s.' % compressed_file)
    with tarfile.open(compressed_file, 'r:gz') as corpus_tar:
        corpus_tar.extractall(path)

    input_file = find_file(path, input_filename)
    target_file = find_file(path, target_filename)

    if input_file and target_file:
        return input_file, target_file

    raise OSError(
        'Download/extraction failed for url %s to path %s' % (url, path)
    )


def txt_line_iterator(path):
    with tf.io.gfile.GFile(path) as f:
        for line in f:
            yield line.strip()


def compile_files(raw_dir, raw_files, tag):
    """
        将多个输入文件和输出文件组合到两个单独的文件中，方便后续处理
    """

    logging.info('Compiling files with tag %s.' & tag)
    filename = '%s-%s' % (_PREFIX, tag)
    input_compiled_file = os.path.join(raw_dir, filename + '.lang1')
    target_compiled_file = os.path.join(raw_dir, filename + '.lang2')

    with tf.io.gfile.GFile(input_compiled_file, mode='w') as input_writer:
        with tf.io.gfile.GFile(target_compiled_file, mode='w') as target_writer:
            for i in range(len(raw_files['inputs'])):
                input_file = raw_files['inputs'][i]
                target_file = raw_files['targets'][i]

                logging.info('Reading files %s and %s.' % (input_file, target_file))
                write_file(input_writer, input_file)
                write_file(target_writer, target_file)
    return input_compiled_file, target_compiled_file


def write_file(writer, filename):
    for line in txt_line_iterator(filename):
        writer.write(line)
        writer.write('\n')


def encode_and_save_files(
        sub_tokenizer_lang_input,
        sub_tokenizer_lang_target,
        data_dir,
        raw_files,
        tag,
        total_shards
):
    """
    :param sub_tokenizer_lang_input: 对输入语料进行分词并转为 ids
    :param sub_tokenizer_lang_target: 对输出语料进行分词并转为 ids
    :param data_dir: 处理后得到的 TFRecord 存储的位置
    :param raw_files: (input_file, target_file)
    :param tag:
    :param total_shards:
    :return:
    """

    # 根据 total_shards 获取多个文件名
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
                 for n in range(total_shards)]
    if all_exist(filepaths):
        logging.info('Files with tag %s already exist.' % tag)
        return filepaths
    logging.info('Saving files with tag %s.' % tag)
    input_file = raw_files[0]
    target_file = raw_files[1]

    tmp_filepaths = [fname + '.incomplete' for fname in filepaths]
    writers = [tf.io.TFRecordWriter(fname) for fname in tmp_filepaths]
    counter, shard = 0, 0
    for counter, (input_line, target_line) in enumerate(
        zip(
            txt_line_iterator(input_file), txt_line_iterator(target_file)
        )
    ):
        if counter > 0 and counter % 100000 == 0:
            logging.info('\tSaving case %d.' % counter)

        example = dict_to_example(
            {
                'inputs': sub_tokenizer_lang_input.encode(input_line, add_eos=True),
                'targets': sub_tokenizer_lang_target.encode(target_line, add_eos=True)
            }
        )
        writers[shard].write(example.SerializaToString())
        shard = (shard + 1) % total_shards

    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.io.gfile.rename(tmp_name, final_name)

    logging.info('Saving %d Examples', counter + 1)

    return filepaths


def shard_filename(path, tag, shard_num, total_shards):
    return os.path.join(
        path, '%s-%s-%.5d-od-%.5d' % (_PREFIX, tag, shard_num, total_shards)
    )


def all_exist(filepaths):
    for fname in filepaths:
        if not tf.io.gfile.exists(fname):
            return False
    return True


def dict_to_example(dictionary):
    features = {}
    for k, v in dictionary.items():
        features[k] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=v)
        )
    return tf.train.Example(features=tf.train.Features(feature=features))
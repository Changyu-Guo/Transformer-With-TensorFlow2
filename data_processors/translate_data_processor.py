# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import tarfile
import collections
import urllib.request
import tensorflow as tf
from absl import app
from absl import logging
from absl import flags
from tokenizers import BertWordPieceTokenizer

sys.path.append('D:\\projects\\Transformers-With-TensorFlow2')

from tokenizations import sub_tokenization


def txt_line_iterator(path):
    """
        打开指定 txt 文件，每次生成一行数据
    """
    with tf.io.gfile.GFile(path) as f:
        for line in f:
            yield line.strip()


def combine_files(save_dir, files, dataset_name, dataset_type):
    """
        当 inputs 和 targets 存在于多个文件中时，可以使用这个函数将数据合并到一个文件内
        save_dir: 合并后的文件存储位置
        files: 应指定为一个字典，包含 inputs 和 targets 键，值对应的则为一个 list，包含所有待合并的文件
               在 list 中，inputs 和 targets 必须对应
        dataset_name: 数据集的名称，必须指定
        dataset_type: 数据集类型，例如 train 或者 eval

        步骤：
            1. 确定合并后的文件名：dataset_name + '-' + dataset_type + '.inputs'/'.targets'
               例如 casict-train
            2. open 两个文件
            3. 循环依次读取 inputs list 和 targets list 中的所有文件
            4. 将 inputs 写入文件，将 targets 写入文件

    """

    inputs_combined_filename = os.path.join(save_dir, dataset_name + '-' + dataset_type + '.inputs')
    targets_combined_filename = os.path.join(save_dir, dataset_name + '-' + dataset_type + '.targets')

    with tf.io.gfile.GFile(inputs_combined_filename, mode='w') as f_inputs:
        with tf.io.gfile.GFile(targets_combined_filename, mode='w') as f_targets:
            for i in range(len(files['inputs'])):
                inputs_file = files['inputs'][i]
                targets_file = files['targets'][i]

                for line in txt_line_iterator(inputs_file):
                    f_inputs.write(line)
                    f_inputs.write('\n')

                for line in txt_line_iterator(targets_file):
                    f_targets.write(line)
                    f_targets.write('\n')

    return inputs_combined_filename, targets_combined_filename


class TranslationFeature:
    def __init__(
            self,
            inputs_ids,
            targets_ids
    ):
        self.inputs_ids = inputs_ids
        self.targets_ids = targets_ids


class FeatureWriter:
    def __init__(self, filename):
        self.filename = filename
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values))
            )
            return feature

        features = collections.OrderedDict()
        features['inputs_ids'] = create_int_feature(feature.inputs_ids)
        features['targets_ids'] = create_int_feature(feature.targets_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def convert_corpus_to_features(
        inputs_tokenizer,
        targets_tokenizer,
        combined_inputs_corpus,
        combined_targets_corpus,
        save_path,
        dataset_name,
        dataset_type,
        shards
):
    """
        将双语语料转为可以作为模型输入的 features

        步骤：
            1. 按照一定的规则生成 shards 个文件名，例如 casict-train-data-001-of-100.tfrecord
            2. 判断文件是否存在，如果存在，直接返回，如果不存在，执行接下来的步骤
            3. 实例化 shards 个 writer
            4. 同时打开 inputs corpus 和 targets corpus
            5. 每次读取 inputs corpus 和 targets corpus 中的一行
            6. 使用 tokenizer 分词并转为 ids
            7. 构造 TranslationFeature
            8. 每个 writer 一行，依次将数据写入不同的 tfrecord 中
    """
    shards_len = len(str(shards))
    template_str = '%s-%s-data-%.{}d-of-%.{}d.tfrecord'.format(shards_len, shards_len)
    tfrecord_paths = [
        os.path.join(
            save_path,
            template_str % (dataset_name, dataset_type, i + 1, shards)
        ) for i in range(shards)
    ]

    all_exist = True
    for path in tfrecord_paths:
        if not tf.io.gfile.exists(path):
            all_exist = False
            break
    if all_exist:
        return tfrecord_paths

    incomplete_tfrecord_paths = [path + '.incomplete' for path in tfrecord_paths]

    tfrecord_writers = [FeatureWriter(path) for path in incomplete_tfrecord_paths]

    shard = 0
    for lines, (inputs_line, targets_line) in enumerate(
            zip(
                txt_line_iterator(combined_inputs_corpus),
                txt_line_iterator(combined_targets_corpus)
            )
    ):
        inputs_ids = inputs_tokenizer.encode(inputs_line, add_special_tokens=False).ids
        targets_ids = targets_tokenizer.encode(targets_line, add_special_tokens=False).ids
        inputs_ids += [inputs_tokenizer.token_to_id('[SEP]')]
        targets_ids = [targets_tokenizer.token_to_id('[CLS]')] + targets_ids + [targets_tokenizer.token_to_id('[SEP]')]

        feature = TranslationFeature(inputs_ids, targets_ids)

        tfrecord_writers[shard].process_feature(feature)
        shard = (shard + 1) % shards

        if lines > 0 and lines % 1000 == 0:
            logging.info('process {} lines'.format(lines))

    for writer in tfrecord_writers:
        writer.close()

    for incomplete_path, path in zip(incomplete_tfrecord_paths, tfrecord_paths):
        tf.io.gfile.rename(incomplete_path, path)

    return tfrecord_paths


def main():
    save_path = 'D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende\\records'
    tf.io.gfile.makedirs(save_path)
    train_inputs_file = 'D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende\\training\\news' \
                        '-commentary-v12.zh-en.zh'
    train_targets_file = 'D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende\\training\\news' \
                         '-commentary-v12.zh-en.en'
    inputs_vocab_file = 'D:\\projects\\Transformers-With-TensorFlow2\\tokenizations\\vocabs\\bert_base_chinese_vocab' \
                        '.txt'
    targets_vocab_file = 'D:\\projects\\Transformers-With-TensorFlow2\\tokenizations\\vocabs' \
                         '\\bert_large_uncased_wwm_vocab.txt'
    inputs_tokenizer = BertWordPieceTokenizer(vocab_file=inputs_vocab_file)
    targets_tokenizer = BertWordPieceTokenizer(vocab_file=targets_vocab_file)

    convert_corpus_to_features(
        inputs_tokenizer,
        targets_tokenizer,
        train_inputs_file,
        train_targets_file,
        save_path=save_path,
        dataset_name='newstest',
        dataset_type='train',
        shards=10
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()

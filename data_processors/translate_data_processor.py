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
                int64_list=tf.train.Int64List(values=list(values))
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
            2. 实例化 shards 个 writer
            3. 同时打开 inputs corpus 和 targets corpus
            4. 每次读取 inputs corpus 和 targets corpus 中的一行
            5. 使用 tokenizer 分词并转为 ids
            6. 构造 TranslationFeature
            7. 每个 writer 一行，依次将数据写入不同的 tfrecord 中
    """
    shards_len = len(str(shards))
    template_str = '%s-%s-data-%.{}d-of-%.{}d.tfrecord'.format(shards_len, shards_len)
    tfrecord_paths = [
        os.path.join(
            save_path,
            template_str % (dataset_name, dataset_type, i, shards)
        ) for i in range(shards)
    ]

    tfrecord_writers = [FeatureWriter(path) for path in tfrecord_paths]

    for lines, (inputs_line, targets_line) in enumerate(
            zip(
                txt_line_iterator(combined_inputs_corpus),
                txt_line_iterator(combined_targets_corpus)
            )
    ):
        inputs_ids = inputs_tokenizer.encode(inputs_line)
        targets_ids = targets_tokenizer.encode(targets_line, add_eos=True)

        feature = TranslationFeature(inputs_ids, targets_ids)

        file_write_fn(feature)


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
        writers[shard].write(example.SerializeToString())
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


def shuffle_records(fname):
    logging.info('Shuffling records in file %s' % fname)

    tmp_fname = fname + '.unshuffled'
    tf.io.gfile.rename(fname, tmp_fname)

    reader = tf.data.TFRecordDataset(tmp_fname)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            logging.info('\tRead: %d', len(records))

    random.shuffle(records)

    with tf.io.TFRecordWriter(fname) as w:
        for count, record in enumerate(records):
            w.write(record)
            if count > 0 and count % 100000 == 0:
                logging.info('\tWriting record: %d', count)

    tf.io.gfile.remove(tmp_fname)


def dict_to_example(dictionary):
    features = {}
    for k, v in dictionary.items():
        features[k] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=v)
        )
    return tf.train.Example(features=tf.train.Features(feature=features))


def make_dir(path):
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)


def define_processor_flags():
    flags.DEFINE_string(
        name='data_dir',
        default='D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende',
        help=''
    )
    flags.DEFINE_string(
        name='raw_dir',
        default='D:\\projects\\Transformers-With-TensorFlow2\\datasets\\translate_ende_raw',
        help=''
    )
    flags.DEFINE_bool(
        name='search',
        default=False,
        help=''
    )


def main(_):
    make_dir(FLAGS.raw_dir)
    make_dir(FLAGS.data_dir)

    train_files_flat = train_files['inputs'] + train_files['targets']
    vocab_file = os.path.join(FLAGS.data_dir, VOCAB_FILE)
    sub_tokenizer = sub_tokenization.Subtokenizer.init_from_files(
        vocab_file,
        train_files_flat,
        _TARGET_VOCAB_SIZE,
        _TARGET_THRESHOLD,
        min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT
    )

    logging.info('Step 4/5: Compiling training and evaluation data')
    compiled_train_files = compile_files(FLAGS.raw_dir, train_files, _TRAIN_TAG)
    compiled_eval_files = compile_files(FLAGS.raw_dir, eval_files, _EVAL_TAG)

    logging.info('Step 5/5: Preprocessing and saving data')
    train_tfrecord_files = encode_and_save_files(
        sub_tokenizer,
        sub_tokenizer,
        FLAGS.data_dir,
        compiled_train_files,
        _TRAIN_TAG,
        _TRAIN_SHARDS
    )
    encode_and_save_files(
        sub_tokenizer,
        sub_tokenizer,
        FLAGS.data_dir,
        compiled_eval_files,
        _EVAL_TAG,
        _EVAL_SHARDS
    )

    # for fname in train_tfrecord_files:
    #     shuffle_records(fname)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_processor_flags()
    FLAGS = flags.FLAGS
    app.run(main)

# -*- coding: utf - 8 -*-

from __future__ import absolute_import

import tensorflow as tf
from data_processors.squad_processor import generate_tf_record_from_json_file


def main():
    train_file_path = 'datasets/train-v1.1.json'
    eval_file_path = 'datasets/dev-v1.1.json'
    vocab_file_path = 'datasets/vocab.txt'
    output_dir = 'test_dir'

    generate_tf_record_from_json_file(
        train_file_path=train_file_path,
        eval_file_path=eval_file_path,
        vocab_file_path=vocab_file_path,
        output_dir=output_dir
    )

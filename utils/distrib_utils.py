# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import json

import tensorflow as tf
from absl import logging


def tpu_initialize(tpu_address):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address
    )
    if tpu_address not in ('', 'local'):
        tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return resolver


def _collective_communication(all_reduce_algorithm):
    collective_communication_options = {
        None: tf.distribute.experimental.CollectiveCommunication.AUTO,
        'ring': tf.distribute.experimental.CollectiveCommunication.RING,
        'nccl': tf.distribute.experimental.CollectiveCommunication.NCCL
    }

    if all_reduce_algorithm not in collective_communication_options:
        raise ValueError(
            'When used with multi_worker_mirrored, valid values for'
            'all_reduce_algorithm are [ring, nccl]. Supplied Value: {}'.format(all_reduce_algorithm)
        )

    return collective_communication_options[all_reduce_algorithm]


def _mirrored_cross_device_ops(all_reduce_algorithm, num_packs):
    if all_reduce_algorithm is None:
        return None

    mirrored_all_reduce_options = {
        'nccl': tf.distribute.NcclAllReduce,
        'hierarchical_copy': tf.distribute.HierarchicalCopyAllReduce
    }

    if all_reduce_algorithm not in mirrored_all_reduce_options:
        raise ValueError(
            'When used with mirrored, valid values for all_reduce_algorithm'
            'are [nccl, hierarchical_copy]. Supplied value: {}'.format(all_reduce_algorithm)
        )

    cross_device_ops_class = mirrored_all_reduce_options[all_reduce_algorithm]
    return cross_device_ops_class(num_packs=num_packs)


def get_distribution_strategy(
        distribution_strategy='mirrored',
        num_gpus=0,
        all_reduce_algorithm=None,
        num_packs=1,
        tpu_address=None
):
    if num_gpus < 0:
        raise ValueError('num_gpus can not be negative')

    distribution_strategy = distribution_strategy.lower()

    # 不使用分布式
    if distribution_strategy == 'off':
        if num_gpus > 1:
            raise ValueError(
                'When {} GPUs are specified, distribution_strategy'
                'flag can not be set to off'.format(num_gpus)
            )
        return None

    # 使用 TPU，需要配置 resolver
    # https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy
    if distribution_strategy == 'tpu':
        resolver = tpu_initialize(tpu_address)
        return tf.distribute.experimental.TPUStrategy(resolver)

    # 多机多卡
    if distribution_strategy == 'multi_worker_mirrored':
        return tf.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=None
        )

    # 单机多卡
    if distribution_strategy == 'mirrored':
        if num_gpus == 0:
            devices = ['device:CPU:0']
        else:
            devices = ['device:GPU:%d' % i for i in range(num_gpus)]

        return tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=_mirrored_cross_device_ops(all_reduce_algorithm, num_packs)
        )

    # 单机单卡
    if distribution_strategy == 'one_device':
        if num_gpus == 0:
            return tf.distribute.OneDeviceStrategy('device:CPU:0')
        if num_gpus > 1:
            raise ValueError(
                'OneDeviceStrategy can not be used for more than '
                'one device'
            )
        return tf.distribute.OneDeviceStrategy('device:GPU:0')

    # 异步训练
    if distribution_strategy == 'parameter_server':
        return tf.distribute.experimental.ParameterServerStrategy()

    raise ValueError(
        'Unrecognized Distribution Strategy: %r' % distribution_strategy
    )


def get_strategy_scope(strategy):
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = None

    return strategy_scope


class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

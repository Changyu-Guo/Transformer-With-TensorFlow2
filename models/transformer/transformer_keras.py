# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
from absl import flags
from absl import logging
from models.transformer import transformerV2
from models.transformer.transformer_params import PARAMS
from optimizations.schedules import WarmupSchedule
from metrics import transformer_metrics
from models.transformer import input_pipeline
from utils import distrib_utils
from tokenizations import sub_tokenization


def evaluate_and_log_bleu(
        model,
        params,
        bleu_source,
        bleu_ref,
        targets_vocab_file,
        distribution_strategy=None
):
    sub_tokenize = sub_tokenization.Subtokenizer(targets_vocab_file)
    uncased_score, cased_score = None


def get_model_params():
    return PARAMS.copy()


class TransformerTask:
    def __init__(self, flags_obj):
        self.flags_obj = flags_obj
        self.predict_model = None

        self.params = params = get_model_params()
        params['num_gpus'] = flags_obj.num_gpus
        params['data_dir'] = flags_obj.data_dir
        params['model_dir'] = flags_obj.model_dir
        params['static_batch'] = flags_obj.static_batch
        params['max_seq_len'] = flags_obj.max_seq_len
        params['decode_batch_size'] = flags_obj.decode_batch_size
        params['decode_max_len'] = flags_obj.decode_max_len
        params['batch_size'] = flags_obj.batch_size or params['batch_size']
        params['repeat_dataset'] = None
        params["enable_tensorboard"] = flags_obj.enable_tensorboard
        params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
        params["steps_between_evaluates"] = flags_obj.steps_between_evals
        params["enable_checkpointing"] = flags_obj.enable_checkpointing

        self.distribution_strategy = distrib_utils.get_distribution_strategy(
            distribution_strategy=flags_obj.distribution_strategy,
            num_gpus=flags_obj.num_gpus,
            all_reduce_algorithm=flags_obj.all_reduce_algorithm,
            num_packs=flags_obj.num_packs
        )

        logging.info('Running transformer with num_gpus = %d' % flags_obj.num_gpus)

        if self.distribution_strategy:
            logging.info(
                'For training, using distribution strategy: %s' % self.distribution_strategy
            )
        else:
            logging.info('Not using any distribution strategy')

    def train(self):
        flags_obj = self.flags_obj
        params = self.params
        _ensure_dir_exist(flags_obj.model_dir)

        # 在 distribution strategy scope 下创建模型和优化器
        # 加载 checkpoint 并 compile
        with distrib_utils.get_strategy_scope(self.distribution_strategy):
            model = transformerV2.create_model(params, is_train=True)
            optimizer = self._create_optimizer()

            current_step = 0
            checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer
            )
            latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
                logging.info('Load checkpoint %s from %s' % (latest_checkpoint, flags_obj.model_dir))
                current_step = optimizer.iterations.numpy()

            model.compile(optimizer)

        model.summary()

        train_dataset = input_pipeline.get_train_dataset(params)

        callbacks = self._create_callbacks()

        cased_score, uncased_score = None, None
        cased_score_history, uncased_score_history = [], []
        while current_step < flags_obj.train_steps:
            remaining_steps = flags_obj.train_steps - current_step
            train_steps_per_eval = (
                remaining_steps if remaining_steps < flags_obj.steps_between_evals
                else flags_obj.steps_between_evaluates
            )
            current_iteration = current_step // flags_obj.steps_between_evaluates

            logging.info('Start train iteration at global step:{}'.format(current_step))

            history = model.fit(
                train_dataset,
                initial_epoch=current_iteration,
                epochs=iteration + 1,
                steps_per_epoch=train_steps_per_eval,
                callbacks=callbacks,
                verbose=1
            )
            current_step += train_steps_per_evaluate
            logging.info('Train history: {}'.format(history.history))

            logging.info('End train iteration at global step: {}'.format(current_step))

            if flags_obj.bleu_score and flags_obj.bleu_ref:
                uncased_score, cased_score = self.eval()
                cased_score_history.append([current_iteration + 1, cased_score])
                uncased_score_history.append([current_iteration + 1, uncased_score])

    def eval(self):
        with distrib_utils.get_strategy_scope(self.distribution_strategy):
            if not self.predict_model:
                self.predict_model = transformerV2.create_model(self.params, False)

            self._load_weights_if_possible(
                self.predict_model,
                tf.train.latest_checkpoint(self.flags_obj.model_dir)
            )
            self.predict_model.summary()

        return evaluate_and_log_bleu(
            self.predict_model,
            self.params,
            self.flags_obj.bleu_source,
            self.flags_obj.bleu_ref,
            self.flags_obj.targets_vocab_file,
            self.distribution_strategy
        )

    def _create_optimizer(self):
        params = self.params
        lr_schedule = WarmupSchedule(
            params['learning_rate'],
            params['hidden_size'],
            params['warmup_steps']
        )
        optimizer = tf.keras.optimizers.Adam(
            lr=lr_schedule,
            beta_1=params['optimizer_adam_beta_1'],
            beta_2=params['optimizer_adam_beta_2'],
            epsilon=params['optimizer_adam_epsilon']
        )
        return optimizer

    def _create_callbacks(self):
        callbacks = []
        if self.flags_obj.enable_tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.flags_obj.model_dir
            )
            callbacks.append(tensorboard_callback)
        if self.flags_obj.enable_checkpointing:
            ckpt_full_path = os.path.join(self.flags_obj.model_dir, 'checkpoint-{04d}.ckpt')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                ckpt_full_path, save_weights_only=True
            )
            callbacks.append(checkpoint_callback)

        return callbacks

    def _load_weights_if_possible(self, model, init_weight_path=None):
        if init_weight_path:
            logging.info('Load weights: {}'.format(init_weight_path))
            model.load_weights(init_weight_path)
        else:
            logging.info('Weights not loaded from path: {}'.format(init_weight_path))


def _ensure_dir_exist(_dir):
    if not tf.io.gfile.exists(_dir):
        tf.io.gfile.makedirs(_dir)


def define_transformer_flags():
    # model checkpoint
    flags.DEFINE_boolean(
        name='enable_checkpoint',
        default=True,
        help='Whether enable checkpoint'
    )
    flags.DEFINE_string(
        name='model_dir',
        default='./models',
        help='Path to save checkpoint files'
    )
    flags.DEFINE_boolean(
        name='clean_model_dir',
        default=False,
        help='If set, model_dir will be removed if it exists'
    )

    # train and evaluate steps
    flags.DEFINE_string(
        name='mode',
        default='train',
        help='train / eval / predict'
    )
    flags.DEFINE_integer(
        name='train_epochs',
        default=1,
        help='train epochs'
    )
    flags.DEFINE_integer(
        name='epochs_between_evaluates',
        default=1,
        help='epochs between evaluates'
    )
    flags.DEFINE_integer(
        name='train_steps',
        default=300000,
        help='train steps'
    )
    flags.DEFINE_integer(
        name='steps_between_evaluates',
        default=5000,
        help='steps between evaluates'
    )
    flags.DEFINE_integer(
        name='validation_steps',
        default=64,
        help='validation steps'
    )
    flags.DEFINE_integer(
        name=''
    )

    # data
    flags.DEFINE_string(
        name='data_dir',
        default='./data',
        help='data dir'
    )
    flags.DEFINE_integer(
        name='batch_size',
        default=32,
        help='global batch size'
    )
    flags.DEFINE_boolean(
        name='static_batch',
        default=False,
        help='static batch'
    )
    flags.DEFINE_integer(
        name='max_seq_len',
        default=256,
        help='max seq len'
    )
    flags.DEFINE_integer(
        name='decode_batch_size',
        default=32,
        help='global decode batch size'
    )
    flags.DEFINE_integer(
        name='decode_max_len',
        default=97,
        help='decode max len'
    )

    # distribution
    flags.DEFINE_string(
        name='distribution_strategy',
        default='mirrored',
        help='distribution strategy'
    )
    flags.DEFINE_integer(
        name='num_gpus',
        default=1,
        help='num gpus'
    )
    flags.DEFINE_string(
        name='all_reduce_algorithm',
        default=None,
        help='all reduce algorithm'
    )
    flags.DEFINE_integer(
        name='num_packs',
        default=1,
        help='num packs'
    )

    # performance
    flags.DEFINE_boolean(
        name='enable_xla',
        default=False,
        help='Whether to enable XLA'
    )

    # history and logs
    flags.DEFINE_boolean(
        name='enable_tensorboard',
        default=False,
        help='Whether to enable Tensorboard callback'
    )
    flags.DEFINE_boolean(
        name='enable_metrics_in_training',
        default=False,
        help='Whether to enable metrics during training'
    )

    # BLEU
    flags.DEFINE_string(
        name='bleu_source',
        default=None,
        help='bleu source'
    )
    flags.DEFINE_string(
        name='bleu_ref',
        default=None,
        help='bleu reference'
    )

    # vocab
    flags.DEFINE_string(
        name='inputs_vocab_file',
        default=None,
        help='vocab file for inputs'
    )
    flags.DEFINE_string(
        name='targets_vocab_file',
        default=None,
        help='vocab file for targets'
    )




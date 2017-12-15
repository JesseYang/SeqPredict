#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from reader import *
from cfgs.config import cfg

TOTAL_BATCH_SIZE = 16


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.feature_len], 'feature'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        feature, label = inputs

        # for layer_name, params in cfg.network:
        for layer_param in cfg.network:
            layer_name = layer_param['name']
            layer_type = layer_name.split('_')[0]

            if layer_type == "fc":
                feature = FullyConnected(layer_name, feature, layer_param['num'])
            elif layer_type == "relu":
                feature = LeakyReLU(layer_name, feature, layer_param['k'])
            elif layer_type == "dp":
                feature = Dropout(layer_name, feature, layer_param['prob'])

        # logits = (LinearWrap(feature)
        #           .FullyConnected('fc_1', 32)
        #           .LeakyReLU('relu_1', 0.1)
        #           .FullyConnected('fc_2', 32)
        #           .LeakyReLU('relu_2', 0.1)
        #           .FullyConnected('fc_3', 64)
        #           .LeakyReLU('relu_3', 0.1)
        #           .FullyConnected('fc_4', 64)
        #           .LeakyReLU('relu_4', 0.1)
        #           .Dropout('dp_2', 0.2)
        #           .FullyConnected('fc_7', 2)())

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wrong = prediction_incorrect(feature, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test):
    is_train = train_or_test == 'train'

    if is_train == True:
        ds = Data(cfg.train_x_f_name, cfg.train_y_f_name)
    else:
        ds = Data(cfg.test_x_f_name, cfg.test_y_f_name)

    ds = BatchData(ds, BATCH_SIZE, remainder=not is_train)
    return ds

def get_config():
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(InferenceRunner(dataset_val, [
                                ClassificationError('wrong-top1', 'val-error-top1')
                                ]), every_k_epochs=3),
            ScheduledHyperParamSetter('learning_rate',
                                     [(0, 1e-2), (30, 3e-3), (60, 1e-3), (90, 3e-4), (120, 1e-4)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help="directory of logging", default=None)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = int(args.batch_size) // NR_GPU

    if args.logdir != None:
        logger.set_logger_dir(os.path.join("train_log", args.logdir))
    else:
        logger.auto_set_dir()
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    AsyncMultiGPUTrainer(config).train()

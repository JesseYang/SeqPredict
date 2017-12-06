#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import os
import numpy as np
from scipy import misc
import argparse

from tensorpack import *

from train import Model
from cfgs.config import cfg

import pdb

def predict(args):


    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["NETWORK_OUTPUT"])

    predict_func = OfflinePredictor(predict_config)
    if os.path.isdir(args.input):
        # this is a directory, should have subdirectories, whose names are numbers and contents are images
        samples = []
        class_dirs = os.listdir(args.input)
        for class_dir in class_dirs:
            klass = int(class_dir)
            class_dir_path = os.path.join(args.input, class_dir)
            for filename in os.listdir(class_dir_path):
                samples.append([klass, os.path.join(class_dir_path, filename)])
        # for (dirpath, dirname, filenames) in os.walk(args.input):
        #     klass = dirpath[len(args.input):]
        #     if klass == "":
        #         continue
        #     for filename in filenames:
        #         samples.append([int(klass), os.path.join(dirpath, filename)])
        wrong = []
        for (idx, sample) in enumerate(samples):
            klass = sample[0]
            img_path = sample[1]
            img = misc.imread(img_path)
            if args.crop == True:
                height, width, _ = img.shape
                y_start = int((height - cfg.img_size) / 2)
                x_start = int((width - cfg.img_size) / 2)
                img = img[y_start:y_start + cfg.img_size, x_start:x_start + cfg.img_size]
            img = np.expand_dims(img, axis=0)

            predictions = predict_func([img])[0]
            result = np.argmax(predictions)
            if klass != result:
                wrong.append([img_path, result])
            if idx % 300 == 299:
                logger.info(str(idx + 1) + "/" + str(len(samples)))
        logger.info("Total number is: " + str(len(samples)))
        logger.info("Error number is: " + str(len(wrong)))
        for wrong_sample in wrong:
            logger.warn("Wrong: " + wrong_sample[0] + ", predicted class is " + str(wrong_sample[1]))

    else:
        # this is a file
        if args.input.endswith(".txt"):
            # this is a txt file
            with open(args.input) as f:
                records = f.readlines()
            records = [e.strip().split(' ') for e in records]
            img_paths = [e[0] for e in records]
            classes = [int(e[1]) for e in records]

            error_stat = np.zeros(cfg.class_num)
            error_detail = { }
            tot_stat = np.zeros(cfg.class_num)

            for img_idx, img_path in enumerate(img_paths):
                if img_idx > 0 and img_idx % 100 == 0:
                    print(img_idx)
                img = misc.imread(img_path)
                if args.crop == True:
                    height, width, _ = img.shape
                    y_start = int((height - cfg.img_size) / 2)
                    x_start = int((width - cfg.img_size) / 2)
                    img = img[y_start:y_start + cfg.img_size, x_start:x_start + cfg.img_size]
                img = np.expand_dims(img, axis=0)
                predictions = predict_func([img])[0]
                result = np.argmax(predictions)

                tot_stat[classes[img_idx]] += 1
                if classes[img_idx] != result:
                    error_stat[classes[img_idx]] += 1
                    if classes[img_idx] not in error_detail.keys():
                        error_detail[classes[img_idx]] = []
                    error_detail[classes[img_idx]].append(result)
            pdb.set_trace()

        else:
            # this should be an image file
            img = misc.imread(args.input)
            if args.crop == True:
                height, width, _ = img.shape
                y_start = int((height - cfg.img_size) / 2)
                x_start = int((width - cfg.img_size) / 2)
                img = img[y_start:y_start + cfg.img_size, x_start:x_start + cfg.img_size]
            img = np.expand_dims(img, axis=0)

            predictions = predict_func([img])[0]
            result = np.argmax(predictions)

            sort_pred = np.sort(predictions)
            sort_idx = np.argsort(predictions)

            logger.info(result)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--input', help='path to the input image or directory or test text file', required=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args = parser.parse_args()
    predict(args)

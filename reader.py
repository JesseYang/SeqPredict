import os, sys
import pdb
import pickle
import numpy as np
from scipy import misc
import random
import six
from six.moves import urllib, range
import copy
import logging
import cv2
import pickle

from tensorpack import *
from cfgs.config import cfg

class Data(RNGDataFlow):

    def __init__(self, x_f_name, y_f_name, shuffle=True):
        self.shuffle = shuffle

        x_f = open(x_f_name, 'rb')
        self.x_data = pickle.load(x_f)

        y_f = open(y_f_name, 'rb')
        self.y_data = pickle.load(y_f)

    def size(self):
        return self.y_data.shape[0]
        # return len(self.imglist)

    def get_data(self):
        idxs = np.arange(self.y_data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # classification
            yield [self.x_data[k], 1 if self.y_data[k] > 0 else 0]

if __name__ == '__main__':
    ds = Data('pred_more_data_alpha_900_10/pred_3/0_train_x',
              'pred_more_data_alpha_900_10/pred_3/0_train_y')
    ds.reset_state()
    prod = ds.get_data()
    dp = next(prod)

    import pdb
    pdb.set_trace()

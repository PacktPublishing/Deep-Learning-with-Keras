# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from time import gmtime, strftime
from keras.callbacks import TensorBoard
import os


def make_tensorboard(set_dir_name='',
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None):
    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir,
                              embeddings_freq=embeddings_freq,
                              embeddings_layer_names=embeddings_layer_names,
                              embeddings_metadata=embeddings_metadata)
    return tensorboard, log_dir

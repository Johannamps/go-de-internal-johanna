# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements the Keras Sequential model."""

import itertools

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.backend import relu, sigmoid
from keras.layers import LSTM

try:
  # Works with Python 3
  from urllib import parse as urlparse
except ImportError:
  # Otherwise use Python 2 verion:
  from urlparse import urlparse


import logging
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

def model_fn(
    n_seq=334, n_features=5, learning_rate=0.003,
    lstm1_nodes=10, lstm2_nodes=10, mlp1_nodes=10
):
    model = models.Sequential()
    model.add(LSTM(lstm1_nodes, input_shape=(n_seq, n_features), return_sequences=True))
    model.add(LSTM(lstm2_nodes))
    model.add(layers.Dense(mlp1_nodes, activation='relu', name='last_layer'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    compile_model(model, learning_rate)
    
    return model

def compile_model(model, learning_rate):
  model.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.SGD(lr=learning_rate),
                metrics=['accuracy'])
  return model

def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()


def generator_input(input_file, chunk_size):
  """Generator function to produce features and labels
     needed by keras fit_generator.
  """

  # The chunk size has to be a multiple of the seq len:
  n_seq = 334
  if chunk_size < n_seq:
    n_sample=10
    chunk_size = n_sample*n_seq
  else:
    n_sample=np.floor(chunk_size/float(n_seq))
    chunk_size = n_sample*n_seq

  X_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
                       chunksize=chunk_size,
                       na_values=" ?")
  y_reader = pd.read_csv(tf.gfile.Open(input_file[1]),
                       chunksize=n_sample,
                       na_values=" ?")

  for X, y in zip(X_reader, y_reader):

    X = X.iloc[:,1:] #  remove useless first column
    y = y.iloc[:,1:] #  remove useless first column
    n_features = X.shape[1]
    X = np.array(X).reshape(X.shape[0]/n_seq, n_seq, n_features)
    y = np.array(y)

    # print(X.shape, y.shape)
    yield (X, y)
    # return ( (input_data.iloc[[index % n_rows]], label.iloc[[index % n_rows]]) for index in itertools.count() )

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

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.backend import relu, softmax
from sklearn.metrics import *

#Python2/3 compatibility imports
from six.moves.urllib import parse as urlparse
from builtins import range

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# csv columns in the input file
CSV_COLUMNS = ('churn_flag', 'timezone_cd', 'tenure_visit', 'branch', 'abc_cd', 'abc_detailed_cd', 'avg_no_main_cat_1y', 'avg_no_main_cat_q1', 'avg_no_main_cat_q2', 'avg_no_main_cat_th', 'avg_no_main_cat_lh', 'avg_articles_6_6', 'avg_articles_q1_q2', 'ret_visits_per_chg_q1_q2', 'ret_sales_per_chg_q1_q2', 'visit_gap_ratio_th', 'visit_gap_ratio_1y', 'visit_gap_ratio_per_chg_6_6', 'avg_days_between_visits', 'home_st_visits_per_q1', 'home_st_visits_per_6_6', 'basket_spend', 'basket_spend_per_chg_6_6', 'basket_spend_per_chg_q1_q2', 'f_sales_per_chg_6_6', 'nf_visits_per_chg_6_6', 'nf_sales_per_chg_6_6', 'p_visits_per_chg_6_6', 'p_sales_per_chg_6_6', 'ret_visits_per_chg_6_6', 'ret_sales_per_chg_6_6', 'f_visits_per_chg_q1_q2', 'nf_visits_per_chg_q1_q2', 'nf_sales_per_chg_q1_q2', 'p_visits_per_chg_q1_q2', 'p_sales_per_chg_q1_q2', 'consistency_mnth_cd', 'consistency_qtr_cd', 'distinct_weeks_bought', 'expect_visit_flag', 'expect_visit_flag2', 'promo_sales_per', 'margin_per', 'food_sales_per', 'food_promo_per', 'nf_promo_per', 'food_colli_per', 'food_pieces_per', 'sales_last_model_period_per', 'visits_last_model_period_per', 'distinct_stores', 'recency', 'margin_1y', 'home_visits_q1', 'home_visits_lh', 'home_visits_th', 'visit_gap_th', 'visit_gap_lh', 'visit_gap_1y', 'promo_sales_1y', 'promo_sales_q1', 'promo_sales_q2', 'promo_sales_th', 'promo_sales_lh', 'nf_promo_sales', 'food_promo_sales', 'promo_visits_q1', 'promo_visits_q2', 'promo_visits_th', 'promo_visits_lh', 'promo_visits_1y', 'nf_sales_1y', 'nf_sales_q1', 'nf_sales_q2', 'nf_sales_th', 'nf_sales_lh', 'nf_visits_q1', 'nf_visits_q2', 'nf_visits_th', 'nf_visits_lh', 'food_sales_1y', 'food_sales_th', 'food_sales_lh', 'food_visits_1y', 'food_visits_q1', 'food_visits_q2', 'pieces', 'food_colli', 'food_pieces', 'colli_1y', 'colli_q1', 'home_store_visits_q1', 'home_store_visits_th', 'home_store_visits_lh', 'visits_q1', 'visits_q2', 'visits_q3', 'visits_q4', 'visits_1y', 'visits_th', 'visits_lh', 'sales_1y', 'sales_q1', 'sales_q2', 'sales_q3', 'sales_q4', 'sales_th', 'sales_lh', 'ret_visits_q1', 'ret_visits_q2', 'ret_visits_th', 'ret_visits_lh', 'ret_sales_q1', 'ret_sales_q2', 'ret_sales_th', 'ret_sales_lh', 'ret_sales_1y', 'prediction_week')


CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]

# Categorical columns with vocab size
# native_country and fnlwgt are ignored
CATEGORICAL_COLS = ()

CONTINUOUS_COLS = ('timezone_cd', 'tenure_visit', 'branch', 'abc_cd', 'abc_detailed_cd', 'avg_no_main_cat_1y', 'avg_no_main_cat_q1', 'avg_no_main_cat_q2', 'avg_no_main_cat_th', 'avg_no_main_cat_lh', 'avg_articles_6_6', 'avg_articles_q1_q2', 'ret_visits_per_chg_q1_q2', 'ret_sales_per_chg_q1_q2', 'visit_gap_ratio_th', 'visit_gap_ratio_1y', 'visit_gap_ratio_per_chg_6_6', 'avg_days_between_visits', 'home_st_visits_per_q1', 'home_st_visits_per_6_6', 'basket_spend', 'basket_spend_per_chg_6_6', 'basket_spend_per_chg_q1_q2', 'f_sales_per_chg_6_6', 'nf_visits_per_chg_6_6', 'nf_sales_per_chg_6_6', 'p_visits_per_chg_6_6', 'p_sales_per_chg_6_6', 'ret_visits_per_chg_6_6', 'ret_sales_per_chg_6_6', 'f_visits_per_chg_q1_q2', 'nf_visits_per_chg_q1_q2', 'nf_sales_per_chg_q1_q2', 'p_visits_per_chg_q1_q2', 'p_sales_per_chg_q1_q2', 'consistency_mnth_cd', 'consistency_qtr_cd', 'distinct_weeks_bought', 'expect_visit_flag', 'expect_visit_flag2', 'promo_sales_per', 'margin_per', 'food_sales_per', 'food_promo_per', 'nf_promo_per', 'food_colli_per', 'food_pieces_per', 'sales_last_model_period_per', 'visits_last_model_period_per', 'distinct_stores', 'recency', 'margin_1y', 'home_visits_q1', 'home_visits_lh', 'home_visits_th', 'visit_gap_th', 'visit_gap_lh', 'visit_gap_1y', 'promo_sales_1y', 'promo_sales_q1', 'promo_sales_q2', 'promo_sales_th', 'promo_sales_lh', 'nf_promo_sales', 'food_promo_sales', 'promo_visits_q1', 'promo_visits_q2', 'promo_visits_th', 'promo_visits_lh', 'promo_visits_1y', 'nf_sales_1y', 'nf_sales_q1', 'nf_sales_q2', 'nf_sales_th', 'nf_sales_lh', 'nf_visits_q1', 'nf_visits_q2', 'nf_visits_th', 'nf_visits_lh', 'food_sales_1y', 'food_sales_th', 'food_sales_lh', 'food_visits_1y', 'food_visits_q1', 'food_visits_q2', 'pieces', 'food_colli', 'food_pieces', 'colli_1y', 'colli_q1', 'home_store_visits_q1', 'home_store_visits_th', 'home_store_visits_lh', 'visits_q1', 'visits_q2', 'visits_q3', 'visits_q4', 'visits_1y', 'visits_th', 'visits_lh', 'sales_1y', 'sales_q1', 'sales_q2', 'sales_q3', 'sales_q4', 'sales_th', 'sales_lh', 'ret_visits_q1', 'ret_visits_q2', 'ret_visits_th', 'ret_visits_lh', 'ret_sales_q1', 'ret_sales_q2', 'ret_sales_th', 'ret_sales_lh', 'ret_sales_1y', 'prediction_week')

LABELS = [1.0, 0.0]
LABEL_COLUMN = 'churn_flag'

#UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
#   list(zip(*CATEGORICAL_COLS))[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    #precision = precision_score(y_true._shape_as_list(), y_pred._shape_as_list())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    #print(y_pred.shape) 
    #print(y_pred._shape_as_list())
    #print(y_pred._shape_tuple())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    print(recall)
    print(type(recall))
    #recall = recall_score(y_true._shape_as_list(), y_pred._shape_as_list())
    return recall

def model_fn(input_dim,
             labels_dim,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.001):
  """Create a Keras Sequential model with layers."""
  model = models.Sequential()

  for units in hidden_units:
    model.add(layers.Dense(units=units,
                           input_dim=input_dim,
                           activation=relu))
    input_dim = units

  # Add a dense final layer with sigmoid function
  model.add(layers.Dense(labels_dim, activation=softmax))
  compile_model(model, learning_rate)
  return model

def compile_model(model, learning_rate):
  model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                metrics=['accuracy', precision, recall])
  print(type(keras.optimizers.RMSprop(lr=learning_rate)), keras.optimizers.RMSprop(lr=learning_rate).get_config())
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

def to_numeric_features(features,feature_cols=None):
  """Convert the pandas input features to numeric values.
     Args:
        features: Input features in the data
          age (continuous)
          workclass (categorical)
          fnlwgt (continuous)
          education (categorical)
          education_num (continuous)
          marital_status (categorical)
          occupation (categorical)
          relationship (categorical)
          race (categorical)
          gender (categorical)
          capital_gain (continuous)
          capital_loss (continuous)
          hours_per_week (continuous)
          native_country (categorical)

        feature_cols: Column list of converted features to be returned.
            Optional, may be used to ensure schema consistency over multiple executions.


  """

  for col in CATEGORICAL_COLS:
    features = pd.concat([features, pd.get_dummies(features[col[0]], drop_first = True)], axis = 1)
    features.drop(col[0], axis = 1, inplace = True)

  # Remove the unused columns from the dataframe
  #for col in UNUSED_COLUMNS:
  #  features.pop(col)

  #Re-index dataframe (in case categories list changed from the previous dataset)
  if feature_cols is not None:
      features = features.T.reindex(feature_cols).T.fillna(0)

  return features

def generator_input(input_file, chunk_size, batch_size=64):
  """Generator function to produce features and labels
     needed by keras fit_generator.
  """

  feature_cols=None
  while True:
      input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
                               names=CSV_COLUMNS,
                               chunksize=chunk_size,
                               na_values=" ?")

      for input_data in input_reader:
        input_data = input_data.dropna()
        label = pd.get_dummies(input_data.pop(LABEL_COLUMN))
        #print('label:', label)

        input_data = to_numeric_features(input_data,feature_cols)

        #Retains schema for next chunk processing
        if feature_cols is None:
            feature_cols=input_data.columns

        idx_len=input_data.shape[0]
        for index in range(0,idx_len,batch_size):
            yield (input_data.iloc[index:min(idx_len,index+batch_size)], label.iloc[index:min(idx_len,index+batch_size)])

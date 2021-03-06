from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib import metrics


tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['churn_flag', 'timezone_cd', 'tenure_visit', 'branch', 'abc_cd',  'key']

LABEL_COLUMN = 'churn_flag'

DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], ['nokey']]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    layers.real_valued_column('timezone_cd'),
    layers.real_valued_column('tenure_visit'),
    layers.real_valued_column('branch'),
    layers.real_valued_column('abc_cd'),
]

def build_estimator(model_dir, hidden_units):
  (accel, pitch, roll, yaw) = INPUT_COLUMNS 
  return tf.contrib.learn.DNNClassifier(
      model_dir=model_dir,
      feature_columns=[accel, pitch, roll, yaw],  # as-is
      hidden_units=hidden_units or [128, 32, 4])


def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders
    )

def generate_csv_input_fn(filename, num_epochs=None, batch_size=512, mode=tf.contrib.learn.ModeKeys.TRAIN):
  def _input_fn():
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    #input_file_names = [filename]

    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)

    value_column = tf.expand_dims(value, -1)

    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)

    features = dict(zip(CSV_COLUMNS, columns))

    label = features.pop(LABEL_COLUMN)

    return features, label

  return _input_fn


def get_eval_metrics():
  return {
     'rmse': tflearn.MetricSpec(metric_fn=metrics.streaming_root_mean_squared_error),
  }

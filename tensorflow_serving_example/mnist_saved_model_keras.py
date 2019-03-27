# Copyright 2016 Google Inc. All Rights Reserved.
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

#! /usr/bin/env python
r"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.

Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    export_dir
"""

from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
# Helper libraries
import numpy as np

tf.app.flags.DEFINE_integer('training_epochs', 5,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 2, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: mnist_saved_model.py [--training_iteration=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_epochs <= 0:
        print('Please specify a positive value for training iteration.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    # Train model
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    output_class = tf.argmax(model.output,axis=-1,name="output_class")

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=FLAGS.training_epochs)
    sess = K.get_session()

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
          tf.compat.as_bytes(export_path_base),
          tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(model.input)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(output_class)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(model.output)

    classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classification_inputs
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classification_outputs_classes,
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                      classification_outputs_scores
              },
              method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': tensor_info_x},
              outputs={'scores': tensor_info_y},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          },
          main_op=tf.tables_initializer(),
          strip_default_attrs=True)

    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
  tf.app.run()

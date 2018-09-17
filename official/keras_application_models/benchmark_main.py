# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark on the keras built-in application models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.keras_application_models import dataset
from official.keras_application_models import model_callbacks
from official.utils.flags import core as flags_core
from official.utils.logs import logger

# Define a dictionary that maps model names to their model classes inside Keras
MODELS = {
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "xception": tf.keras.applications.Xception,
    "resnet50": tf.keras.applications.ResNet50,
    "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
    "mobilenet": tf.keras.applications.MobileNet,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    # TODO(b/80431378)
    # "nasnetlarge": tf.keras.applications.NASNetLarge,
    # "nasnetmobile": tf.keras.applications.NASNetMobile,
}


def run_keras_model_benchmark(_):
  """Run the benchmark on keras model."""
  # Ensure a valid model name was supplied via command line argument
  if FLAGS.model not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
                         "be a key in the `MODELS` dictionary.")

  # Load the model
  tf.logging.info("Benchmark on {} model...".format(FLAGS.model))
  keras_model = MODELS[FLAGS.model]
  model = keras_model(weights=None)

  # Get dataset
  dataset_name = "ImageNet"
  if FLAGS.use_synthetic_data:
    tf.logging.info("Using synthetic dataset...")
    dataset_name += "_Synthetic"
    train_num_images = FLAGS.batch_size
    val_num_images = FLAGS.batch_size
    train_dataset = dataset.generate_synthetic_input_dataset(
        FLAGS.model, train_num_images)
    val_dataset = dataset.generate_synthetic_input_dataset(
        FLAGS.model, val_num_images)
  else:
    raise ValueError("Only synthetic dataset is supported!")

  # If run with multiple GPUs
  num_gpus = flags_core.get_num_gpus(FLAGS)
  if num_gpus > 0:
    model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)

  # Configure the model
  model.compile(loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

  # Create benchmark logger for benchmark logging
  run_params = {
      "batch_size": FLAGS.batch_size,
      "synthetic_data": FLAGS.use_synthetic_data,
      "train_epochs": FLAGS.train_epochs
  }

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name=FLAGS.model,
      dataset_name=dataset_name,
      run_params=run_params,
      test_id=FLAGS.benchmark_test_id)

  # Create callbacks that log metric values about the training and evaluation
  callbacks = model_callbacks.get_model_callbacks(
      FLAGS.callbacks,
      batch_size=FLAGS.batch_size,
      metric_logger=benchmark_logger)
  # Train and evaluate the model
  history = model.fit(
      train_dataset,
      epochs=FLAGS.train_epochs,
      callbacks=callbacks,
      validation_data=val_dataset,
      steps_per_epoch=int(np.ceil(train_num_images / FLAGS.batch_size)),
      validation_steps=int(np.ceil(val_num_images / FLAGS.batch_size))
  )

  tf.logging.info("Logging the evaluation results...")
  for epoch in range(FLAGS.train_epochs):
    eval_results = {
        "accuracy": history.history["val_acc"][epoch],
        "loss": history.history["val_loss"][epoch],
        tf.GraphKeys.GLOBAL_STEP: (epoch + 1) * np.ceil(
            train_num_images/FLAGS.batch_size)
    }
    benchmark_logger.log_evaluation_result(eval_results)

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def define_keras_benchmark_flags():
  """Add flags for keras built-in application models."""
  flags_core.define_base(hooks=False)
  flags_core.define_performance()
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      data_format="channels_last",
      use_synthetic_data=True,
      batch_size=32,
      train_epochs=2)

  flags.DEFINE_enum(
      name="model", default=None,
      enum_values=MODELS.keys(), case_sensitive=False,
      help=flags_core.help_wrap(
          "Model to be benchmarked."))

  flags.DEFINE_list(
      name="callbacks",
      default=["ExamplesPerSecondCallback", "LoggingMetricCallback"],
      help=flags_core.help_wrap(
          "A list of (case insensitive) strings to specify the names of "
          "callbacks. For example: `--callbacks ExamplesPerSecondCallback,"
          "LoggingMetricCallback`"))


def main(_):
  with logger.benchmark_context(FLAGS):
    run_keras_model_benchmark(FLAGS)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_keras_benchmark_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)

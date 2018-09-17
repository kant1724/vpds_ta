"""The main module for sentiment analysis.

The model makes use of concatenation of two CNN layers
with different kernel sizes.
See `sentiment_model.py` for more details about the models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
try:
    from official.utils.misc import distribution_utils
    from detection.models.a_cnn.model import sentiment_model
    from detection.models.a_cnn.model import model_manager
    import tensorflow as tf
    import os
except:
    print('error importing')
from detection.models.a_cnn import properties as p

class runner():
    root = p.get_root()
    estimator = None
    EMBEDDING_DIM = 256
    VOCABULARY_SIZE = 6000
    SENTENCE_LENGTH = 200
    CNN_FILTERS = 512
    DROPOUT_RATE = 0.7
    TRAIN_EPOCHS = 200
    BATCH_SIZE = 30
    HOOKS = ""
    MODEL_DIR = None
    NUM_GPUS = 1
    NUM_CLASS = 2
    EPOCHS_BETWEEN_EVALS = 1
    BENCHMARK_TEST_ID = None
    
    def init(self, user, project, data_type, slice_type):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("Loading the dataset...")
        
        self.MODEL_DIR = os.path.join(p.get_root(), user, project, data_type, slice_type, p.get_working_directory())
        
        keras_model = sentiment_model.CNN(
            self.EMBEDDING_DIM, self.VOCABULARY_SIZE,
            self.SENTENCE_LENGTH,
            self.CNN_FILTERS, self.NUM_CLASS, self.DROPOUT_RATE)
        
        tf.logging.info("Creating Estimator from Keras model...")
        self.estimator = self.convert_keras_to_estimator(
            keras_model, self.NUM_GPUS, self.MODEL_DIR)
        
    def predict(self, train_enc_ids):
        train_input_fn, _ = model_manager.construct_input_fns(
            train_enc_ids, ['0'], self.VOCABULARY_SIZE,
            self.SENTENCE_LENGTH, self.BATCH_SIZE, repeat=self.EPOCHS_BETWEEN_EVALS)
        predict_result = self.estimator.predict(input_fn=train_input_fn)
        for result in predict_result:
            return int(round(result['dense'][1] * 100))

    def convert_keras_to_estimator(self, keras_model, num_gpus, model_dir=None):
        keras_model.compile(optimizer="rmsprop",
                            loss="categorical_crossentropy", metrics=["accuracy"])
        
        distribution = distribution_utils.get_distribution_strategy(
            num_gpus, all_reduce_alg=None)
        run_config = tf.estimator.RunConfig(train_distribute=distribution)
        
        estimator = tf.keras.estimator.model_to_estimator(
            keras_model=keras_model, model_dir=model_dir, config=run_config)
        
        return estimator
    
    def get_estimator(self):
        return self.estimator
    
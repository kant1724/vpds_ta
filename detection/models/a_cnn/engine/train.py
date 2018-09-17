"""The main module for sentiment analysis.

The model makes use of concatenation of two CNN layers
with different kernel sizes.
See `sentiment_model.py` for more details about the models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
try:
    from official.utils.logs import hooks_helper
    from official.utils.logs import logger
    from official.utils.misc import distribution_utils
    from detection.models.a_cnn.model import sentiment_model
    from detection.models.a_cnn.model import model_manager
    
    import tensorflow as tf
except:
    print('error importing')

import os
from detection.models.a_cnn import properties as p

class trainer():
    root = p.get_root()
    EMBEDDING_DIM = 256
    VOCABULARY_SIZE = 6000
    SENTENCE_LENGTH = 200
    CNN_FILTERS = 512
    DROPOUT_RATE = 0.7
    TRAIN_EPOCHS = 20000
    BATCH_SIZE = 30
    HOOKS = ""
    MODEL_DIR = None         
    NUM_GPUS = 1
    NUM_CLASS = 2
    EPOCHS_BETWEEN_EVALS = 1
    BENCHMARK_TEST_ID = None
    
    stop_yn = False
    slice_type = None
    
    def train(self, user, project, data_type, slice_type, end_step, train_enc_ids, train_dec):
        self.stop_yn = False
        
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("Loading the dataset...")
        
        self.slice_type = slice_type
        self.MODEL_DIR = os.path.join(p.get_root(), user, project, data_type, slice_type, p.get_working_directory())
        if os.path.isdir(self.MODEL_DIR) == False:
            os.makedirs(self.MODEL_DIR)

        train_input_fn, eval_input_fn = model_manager.construct_input_fns(
            train_enc_ids, train_dec, self.VOCABULARY_SIZE,
            self.SENTENCE_LENGTH, self.BATCH_SIZE, repeat=self.EPOCHS_BETWEEN_EVALS)
        
        keras_model = sentiment_model.CNN(
            self.EMBEDDING_DIM, self.VOCABULARY_SIZE,
            self.SENTENCE_LENGTH,
            self.CNN_FILTERS, self.NUM_CLASS, self.DROPOUT_RATE)
        
        tf.logging.info("Creating Estimator from Keras model...")
        estimator = self.convert_keras_to_estimator(
            keras_model, self.NUM_GPUS, self.MODEL_DIR)
        
        train_hooks = hooks_helper.get_train_hooks(
            self.HOOKS,
            batch_size=self.BATCH_SIZE
        )
        run_params = {
            "batch_size": self.BATCH_SIZE,
            "train_epochs": self.TRAIN_EPOCHS,
        }
        benchmark_logger = logger.get_benchmark_logger()
        benchmark_logger.log_run_info(
            model_name="vpds_analysis",
            dataset_name="vpds",
            run_params=run_params,
            test_id=self.BENCHMARK_TEST_ID)
        
        total_training_cycle = self.TRAIN_EPOCHS\
          // self.EPOCHS_BETWEEN_EVALS
        
        self.clear_training_message(user, project, data_type, slice_type, end_step)
        
        end_step = int(end_step)
        for cycle_index in range(total_training_cycle):
            tf.logging.info("Starting a training cycle: {}/{}".format(
                cycle_index + 1, total_training_cycle))
            
            estimator.train(input_fn=train_input_fn, hooks=train_hooks)
            
            if self.stop_yn:
                break
            
            eval_results = estimator.evaluate(input_fn=eval_input_fn)
            
            benchmark_logger.log_evaluation_result(eval_results)
            tf.logging.info("Iteration {}".format(eval_results))
            self.make_training_message(user, project, data_type, slice_type, eval_results, end_step)
            
            if self.stop_yn or end_step <= int(str(eval_results['global_step'])):
                break
            
        tf.keras.backend.clear_session()
    
    def convert_keras_to_estimator(self, keras_model, num_gpus, model_dir=None):
        keras_model.compile(optimizer="rmsprop",
                            loss="categorical_crossentropy", metrics=["accuracy"])
        
        distribution = distribution_utils.get_distribution_strategy(
            num_gpus, all_reduce_alg=None)
        run_config = tf.estimator.RunConfig(train_distribute=distribution)
        
        estimator = tf.keras.estimator.model_to_estimator(
            keras_model=keras_model, model_dir=model_dir, config=run_config)
        
        return estimator
    
    def clear_training_message(self, user, project, data_type, slice_type, end_step):
        with open(os.path.join(self.root, user, project, data_type, slice_type, 'working_dir', 'training_info.txt'), 'w', encoding='utf8') as f1:
            f1.write('')
        with open(os.path.join(self.root, user, project, data_type, slice_type, 'working_dir', 'end_step.txt'), 'w', encoding='utf8') as f2:
            f2.write(str(end_step))
        
    def make_training_message(self, user, project, data_type, slice_type, eval_results, end_step):
        accuracy = str(eval_results['accuracy'])
        loss = str(eval_results['loss'])
        global_step = str(eval_results['global_step'])
        with open(os.path.join(self.root, user, project, data_type, slice_type, 'working_dir', 'training_info.txt'), 'w', encoding='utf8') as f1: 
            f1.write(accuracy + "," + loss + "," + global_step)
        with open(os.path.join(self.root, user, project, data_type, slice_type, 'working_dir', 'end_step.txt'), 'w', encoding='utf8') as f2:
            f2.write(str(end_step))
            
    def stop(self):
        self.stop_yn = True
        
import os
from detection.models.a_cnn.engine import run
from detection.models.a_cnn import properties as p
from detection.models.a_cnn import running_thread as rt
from detection.models.a_cnn import training_thread as tt
from detection.models.a_cnn.file_processor import training_file_creator
from detection.models.a_cnn.file_processor import training_file_manager 
from detection.models.a_cnn.model import model_manager
from detection.extension import tokenization

def get_answer(user, project, data_type, slice_type, x, cnn_training_target):    
    x = tokenization.get_tokenized_text_to_train([x], cnn_training_target)[0]
    predict_result = rt.get_answer(user, project, data_type, slice_type, x)
    
    return str(predict_result)

def start_training(user, project, data_type, slice_type, end_step, language):
    root = p.get_root()
    train_enc_ids, train_dec_ids, train_enc, train_dec = training_file_creator.prepare_custom_data(root, user, project, data_type, slice_type, 1000, 5000, language)
    tt.start_training_thread(user, project, data_type, slice_type, end_step, train_enc_ids, train_dec)    

def upload_training_data(user, project, data_type, slice_type, x, y, cnn_training_target):
    root = p.get_root()
    x = tokenization.get_tokenized_text_to_train(x, cnn_training_target)

    training_file_manager.upload_training_data(root, user, project, data_type, slice_type, x, y)
    
def stop_training(user, project, data_type): 
    tt.stop_training_thread(user, project, data_type)    

def get_training_info(user, project, data_type):
    root = p.get_root()
    trainer = tt.get_trainer_object(user, project, data_type)
    with open(os.path.join(root, user, project, data_type, trainer.slice_type, 'working_dir', 'training_info.txt'), 'r', encoding='utf8') as f1:
        with open(os.path.join(root, user, project, data_type, trainer.slice_type, 'working_dir', 'end_step.txt'), 'r', encoding='utf8') as f2:
            training_info = f1.readline()
            end_step = f2.readline()
            return training_info, end_step

def is_training(user, project, data_type): 
    res = tt.get_trainer_thread(user, project, data_type)
    is_training_yn = 'N'
    if res != None:
        is_training_yn = 'Y'
        
    return is_training_yn

def delete_ckpt(user, project, data_type, slice_type):
    model_manager.delete_ckpt(user, project, data_type, slice_type)

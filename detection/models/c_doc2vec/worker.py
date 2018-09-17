import os
from detection.models.c_doc2vec import running_thread as rt
from detection.models.c_doc2vec import training_thread as tt
from detection.models.c_doc2vec import properties as p
from detection.models.c_doc2vec import uploader

def get_answer(user, project, data_type, x):    
    predict_result, similar_sample, tokenized = rt.get_answer(p.get_root(), user, project, data_type, x)
    
    return str(predict_result), str(similar_sample), str(tokenized)

def upload_training_data(user, project, data_type, vp_data, voca_list):
    rt.remove_runner()
    uploader.upload_vp_data(p.get_root(), user, project, data_type, vp_data, voca_list)

def start_training(user, project, data_type, end_step):
    tt.start_training_thread(user, project, data_type, end_step)    

def get_uploading_config(user, project, data_type):
    root = p.get_root()
    with open(os.path.join(root, user, project, data_type, 'vp_data_file', 'uploading_config.txt'), 'r', encoding='utf8') as f:
        uploading_config = f.readline()

        return uploading_config

def is_training(user, project, data_type): 
    res = tt.get_trainer_object(user, project, data_type)
    is_training_yn = 'N'
    if res != None:
        end_yn = res.is_end()
        if end_yn == 'Y':
            tt.remove_training_thread(user, project, data_type)
        else:
            is_training_yn = 'Y'
        
    return is_training_yn

def get_training_info(user, project, data_type):
    res = tt.get_trainer_object(user, project, data_type)
    if res != None:
        return '', ''
    
    return '', ''

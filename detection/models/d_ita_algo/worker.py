import os
import datetime
from detection.models.d_ita_algo import running_thread as rt
from detection.models.d_ita_algo import training_thread as tt

from detection.models.d_ita_algo import properties as p
from detection.models.d_ita_algo import uploader

def get_answer(user, project, data_type, x):    
    predict_result, similar_sample, tokenized = rt.get_answer(p.get_root(), user, project, data_type, x)
    
    return str(predict_result), str(similar_sample), str(tokenized)

def upload_vp_data(user, project, data_type, vp_data, voca_list):
    rt.remove_runner(user, project, data_type)
    uploader.upload_vp_data(p.get_root(), user, project, data_type, vp_data, voca_list)

def start_training(user, project, data_type, end_step):
    tt.start_training_thread(user, project, data_type, end_step)    

def get_training_info(user, project, data_type):
    res = tt.get_trainer_object(user, project, data_type)
    if res != None:
        return '', ''
    
    return '', ''

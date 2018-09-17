import os
from detection.models.b_jaro_winkler import running_thread as rt
from detection.models.b_jaro_winkler import properties as p
from detection.models.b_jaro_winkler import uploader

def get_answer(user, project, data_type, x
             , min_vp_voca_same_rate, vp_threshold, less_threshold_decrease_point):    
    predict_result, similar_sample, tokenized = rt.get_answer(p.get_root(), user, project, data_type, x
                                                            , min_vp_voca_same_rate, vp_threshold, less_threshold_decrease_point)
    
    return str(predict_result), str(similar_sample), str(tokenized)

def upload_vp_data(user, project, data_type, vp_data, voca_list, uploading_config):
    rt.remove_runner(user, project, data_type)
    uploader.upload_vp_data(p.get_root(), user, project, data_type, vp_data, voca_list, uploading_config)

def get_uploading_config(user, project, data_type):
    root = p.get_root()
    with open(os.path.join(root, user, project, data_type, 'vp_data_file', 'uploading_config.txt'), 'r', encoding='utf8') as f:
        uploading_config = f.readline()

        return uploading_config

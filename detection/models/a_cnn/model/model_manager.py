import glob
import shutil
import os
import numpy as np
from detection.models.a_cnn.data.util import pad_sentence, to_dataset, START_CHAR, OOV_CHAR
from detection.models.a_cnn import properties as p

NUM_CLASS = 2

def construct_input_fns(train_enc_ids, train_dec_ids, vocabulary_size, sentence_length,
                        batch_size, repeat=1):
    x_train = []
    for t_id in train_enc_ids:
        t_id_arr = t_id.split(" ")
        t_id_arr_int = []
        for t_id in t_id_arr:
            t_id_arr_int.append(int(t_id))
        x_train.append(t_id_arr_int)
        
    y_train = []
    for tdi in train_dec_ids:
        y_train.append(int(tdi[0]))

    def train_input_fn():      
        dataset = to_dataset(
            np.array([pad_sentence(s, sentence_length) for s in x_train]),
            np.eye(NUM_CLASS)[y_train], batch_size, repeat)
        dataset = dataset.shuffle(len(x_train), reshuffle_each_iteration=True)
        return dataset
    
    def eval_input_fn():
        dataset = to_dataset(
            np.array([pad_sentence(s, sentence_length) for s in x_train]),
            np.eye(NUM_CLASS)[y_train], batch_size, repeat)
        return dataset
    
    return train_input_fn, eval_input_fn

def delete_ckpt(user, project, data_type, slice_type):
    root = p.get_root()
    path = os.path.join(root, user, project, data_type, slice_type, p.get_working_directory())    
    filelist = glob.glob(os.path.join(path, '*'))    
    for file in filelist:
        try:
            eval_path = os.path.join(path, 'eval')
            if str(file) == os.path.join(path, 'eval'):
                eval_list = glob.glob(os.path.join(eval_path, '*'))
                for eval_file in eval_list:
                    os.unlink(eval_file)
                os.rmdir(file)
            else:
                os.unlink(file)
        except:
            print("exception occurs in file : " + str(file))
    training_info_path = os.path.join(path, 'training_info.txt')
    end_step_path = os.path.join(path, 'end_step.txt')
    with open(training_info_path, 'w', encoding='utf8') as f:
        f.write('')
    with open(end_step_path, 'w', encoding='utf8') as f:
        f.write('')

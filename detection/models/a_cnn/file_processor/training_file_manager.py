import os

def get_answer_and_answer_num(user, project):
    _, _, ad = get_training_data(user, project)
    answer_dict = {}
    for line in ad:
        l = line.replace('\n', '')
        ll = l.split('^')
        answer_dict[ll[1]] = ll[0]
        
    return answer_dict

def upload_training_data(root, user, project, data_type, slice_type, training_text, vp_yn):
    path = os.path.join(root, user, project, data_type, slice_type, 'training_files')
    if os.path.isdir(path) == False:
        os.makedirs(path)
        
    fw1 = open(os.path.join(path, 'train.enc'), 'w', encoding='utf8')
    fw2 = open(os.path.join(path, 'train.dec'), 'w', encoding='utf8')
    
    for t in training_text:
        fw1.write(t.replace('\n', ' ') + '\n')
        
    for v in vp_yn:
        fw2.write(v.replace('\n', ' ') + '\n')
        
    return ''

def get_training_data(root, user, project, data_type, slice_type):
    path = os.path.join(root, user, project, data_type, slice_type, 'training_files')
    
    with open(os.path.join(path, 'train.enc'), 'r', encoding='utf8') as f1:
        with open(os.path.join(path, 'train.dec'), 'r', encoding='utf8') as f2:
            train_enc = []
            train_dec = []
            lines1 = f1.readlines()
            for line in lines1:
                train_enc.append(line.replace('\n', ''))
            lines2 = f2.readlines()
            for line in lines2:
                train_dec.append(line.replace('\n', ''))

    return train_enc, train_dec

import os
from detection.extension import tokenization

def upload_vp_data(root, user, project, data_type, vp_data, voca_list, uploading_config):
    path = os.path.join(root, user, project, data_type, 'vp_data_file')
    if os.path.isdir(path) == False:
        os.makedirs(path)
    
    vp_yn_dict = {}
    
    with open(os.path.join(path, 'voca_data.txt'), 'w', encoding='utf8') as fw:
        for voca in voca_list:
            voca_nm = voca['voca_nm']
            voca_entity = voca['voca_entity']
            vp_yn = voca['vp_yn']
            voca_weight = voca['voca_weight']
            fw.write(voca_nm + "^" + voca_entity + "^" + vp_yn + "^" + str(voca_weight) + "\n")
            vp_yn_dict[voca_nm] = vp_yn
    
    with open(os.path.join(path, 'tokenized_vp_data.txt'), 'w', encoding='utf8') as fw1:
        with open(os.path.join(path, 'vp_data_nouns.txt'), 'w', encoding='utf8') as fw2:
            for vp_text in vp_data:
                nouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(vp_text), vp_yn_dict)
                fw1.write(" ".join(tokenized) + '\n')
                fw2.write(" ".join(nouns) + '\n')
    
    with open(os.path.join(path, 'uploading_config.txt'), 'w', encoding='utf8') as fw:
        fw.write(uploading_config)

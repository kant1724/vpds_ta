import os
from detection.extension import tokenization
import shutil

def upload_vp_data(root, user, project, data_type, vp_data, voca_list):
    path = os.path.join(root, user, project, data_type, 'vp_data_file')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    vp_yn_dict = {}
    voca_entity_dict = {}
    voca_weight_dict = {}
    group = {}
    voca_group = {}
    
    for i in range(len(voca_list)):
        group_no = voca_list[i]['group_no']
        if voca_group.get(group_no, '') != '':
            voca_group[group_no].append(voca_list[i])
        else:
            voca_group[group_no] = [voca_list[i]]
        
    for i in range(len(vp_data)):
        group_no = vp_data[i][0]
        if group.get(group_no, '') != '':
            group[group_no].append(vp_data[i][1])
        else:
            group[group_no] = [vp_data[i][1]]
    
    for key in group:
        group_dir = os.path.join(path, str(key))
        if voca_group.get(key, None) == None:
            voca_group[key] = [{'voca_nm' : 'Empty', 'voca_entity' : '', 'vp_yn' : 'N', 'voca_weight' : '1', 'group_no' : key}]
        if os.path.isdir(group_dir) == False:
            os.makedirs(group_dir)
        with open(os.path.join(group_dir, 'voca_data.txt'), 'w', encoding='utf8') as fw:
            vp_yn_dict[key] = {}
            voca_entity_dict[key] = {}
            voca_weight_dict[key] = {}
            v_list = voca_group[key]
            for voca in v_list:
                voca_nm = voca['voca_nm']
                voca_entity = voca['voca_entity']
                vp_yn = voca['vp_yn']            
                voca_weight = voca['voca_weight']
                group_no = voca['group_no']
                fw.write(voca_nm + "^" + voca_entity + "^" + vp_yn + "^" + str(voca_weight) + "^" + str(group_no) + "\n")
                vp_yn_dict[key][voca_nm] = vp_yn
                voca_entity_dict[key][voca_nm] = voca_entity
                voca_weight_dict[key][voca_nm] = voca_weight
    
    for key in group:
        group_dir = os.path.join(path, str(key))        
        if os.path.isdir(group_dir) == False:
            os.makedirs(group_dir)
        with open(os.path.join(group_dir, 'tokenized_vp_data.txt'), 'w', encoding='utf8') as fw1:
            with open(os.path.join(group_dir, 'vp_data_nouns.txt'), 'w', encoding='utf8') as fw2:
                with open(os.path.join(group_dir, 'vp_data_tagged.txt'), 'w', encoding='utf8') as fw3:
                        vp_data_in_group = group[key]
                        for i in range(len(vp_data_in_group)):
                            vp_text = vp_data_in_group[i]
                            nouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(vp_text), vp_yn_dict[key])
                            fw1.write(" ".join(tokenized) + '\n')
                            fw2.write(" ".join(nouns) + '\n')
                            fw3.write(" ".join(tokenization.tagging_words(nouns, voca_entity_dict[key])) + '\n')        

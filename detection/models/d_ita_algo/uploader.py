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
    with open(os.path.join(path, 'voca_data.txt'), 'w', encoding='utf8') as fw:
        for voca in voca_list:
            voca_nm = voca['voca_nm']
            voca_entity = voca['voca_entity']
            vp_yn = voca['vp_yn']            
            voca_weight = voca['voca_weight']
            fw.write(voca_nm + "^" + voca_entity + "^" + vp_yn + "^" + str(voca_weight) + "\n")
            vp_yn_dict[voca_nm] = vp_yn
            voca_entity_dict[voca_nm] = voca_entity
            voca_weight_dict[voca_nm] = voca_weight
    group = {}
    for i in range(len(vp_data)):
        group_no = vp_data[i][0]
        if group.get(group_no, '') != '':
            group[group_no].append(vp_data[i][1])
        else:
            group[group_no] = [vp_data[i][1]]
    
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
                        nouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(vp_text), vp_yn_dict)
                        nnouns = []
                        for n in nouns:
                            if voca_weight_dict.get(n, None) != None:
                                nnouns.append(n)
                        fw1.write(" ".join(tokenized) + '\n')
                        fw2.write(" ".join(nnouns) + '\n')
                        fw3.write(" ".join(tokenization.tagging_words(nnouns, voca_entity_dict)) + '\n')
                        
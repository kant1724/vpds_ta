from detection.extension import connect_moara as cm 

def get_tokenized_text_to_train(vp_data_list, cnn_training_target):
    res = []
    for vp_data in vp_data_list:
        vp_text = vp_data.replace('^', ' ')
        if cnn_training_target == '1': 
            tokenized_text = cm.morphs(vp_text)
        elif cnn_training_target == '2':
            tokenized_text = cm.nouns(vp_text)
        res.append(" ".join(tokenized_text))
        
    return res

def pos(text):
    return cm.pos(text)

def extract_vp_word_in_pos(pos, vp_yn):
    start = 0
    len_res = len(pos)
    nouns = []
    tokenized = []
    while start < len_res:
        if start + 2 < len_res:
            three_part = pos[start][0] + pos[start + 1][0] + pos[start + 2][0]
            if vp_yn.get(three_part, None) != None:
                tokenized.append(three_part)
                nouns.append(three_part)
                start += 3                
                continue
        if start + 1 < len_res:
            two_part = pos[start][0] + pos[start + 1][0]
            if vp_yn.get(two_part, None) != None:
                tokenized.append(two_part)
                nouns.append(two_part)
                start += 2
                continue
        first = pos[start][0]
        t = pos[start][1]
        tokenized.append(first)
        if t == 'NNP' or t == 'NNG':
            if len(first) > 1:
                nouns.append(first)
        start += 1
    
    return nouns, tokenized

def tagging_words(words, voca_entity):
        replaced = []
        for i in range(len(words)):
            entity = voca_entity.get(words[i], '')
            new_word = words[i]
            if entity != '':
                new_word = entity 
            replaced.append(new_word)

        return replaced

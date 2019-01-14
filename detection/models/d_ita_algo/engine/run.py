import os
import time
from detection.extension import tokenization
from detection.models.d_ita_algo.engine import ita_algo
import gensim.models as g

class runner():
    vp_yn = {}
    voca_entity = {}
    vp_data = []
    vp_data_nouns = {}
    vp_data_with_tag = {}
    vp_data_tokenized = {}
    vp_yn_idx = {}
    ready_to_predict = False
    error = False
    SLEEP_TIME = 0.5
    retry_limit = 200
    doc2vec_model = {}
    
    def init(self, root, user, project, data_type):
        self.vp_yn = {}
        self.voca_weight = {}
        self.voca_entity = {}
        self.vp_data_nouns = {}
        self.vp_data_with_tag = {}
        self.vp_data_tokenized = {}
        self.vp_yn_idx = {}
        self.ready_to_predict = False
        self.error = False
        path = os.path.join(root, user, project, data_type, 'vp_data_file')
        self.error = False
        group_list = os.listdir(path)
        for group_no in group_list:
            with open(os.path.join(path, group_no, 'voca_data.txt'), 'r', encoding='utf8') as f:
                vp_yn_in_group = {}
                voca_entity_in_group = {}
                voca_weight_in_group = {}
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    arr = line.split('^')
                    voca_entity_in_group[arr[0]] = arr[1]
                    vp_yn_in_group[arr[0]] = arr[2]
                    voca_weight_in_group[arr[0]] = arr[3]
            self.vp_yn[group_no] = vp_yn_in_group
            self.voca_entity[group_no] = voca_entity_in_group
            self.voca_weight[group_no] = voca_weight_in_group
        group_list = os.listdir(path)
        for group_no in group_list:
            if os.path.isdir(os.path.join(path, group_no)) == False:
                continue
            vp_data_nouns_in_group = []
            vp_data_with_tag_in_group = []
            vp_data_tokenized_in_group = []
            vp_yn_idx_in_group = {}
            with open(os.path.join(path, group_no, 'vp_data_nouns.txt'), 'r', encoding='utf8') as f1:
                with open(os.path.join(path, group_no, 'tokenized_vp_data.txt'), 'r', encoding='utf8') as f2:
                    vp_data_idx = 0
                    lines = f1.readlines()
                    for line in lines:
                        line = line.replace('\n', '')
                        nouns = line.split(' ')
                        for i in range(len(nouns)):
                            if self.vp_yn[group_no].get(nouns[i], '') == 'Y':
                                if vp_yn_idx_in_group.get(nouns[i], '') != '':
                                    if vp_data_idx not in vp_yn_idx_in_group[nouns[i]]:
                                        vp_yn_idx_in_group[nouns[i]].append(vp_data_idx)
                                else:
                                    vp_yn_idx_in_group[nouns[i]] = [vp_data_idx]
                        vp_data_nouns_in_group.append(nouns)
                        vp_data_with_tag_in_group.append(tokenization.tagging_words(nouns, self.voca_entity))
                        vp_data_idx += 1
                    lines = f2.readlines()
                    for line in lines:
                        line = line.replace('\n', '')
                        tokenized = line.split(' ')
                        vp_data_tokenized_in_group.append(tokenized)
            self.vp_data_nouns[group_no] = vp_data_nouns_in_group
            self.vp_data_with_tag[group_no] = vp_data_with_tag_in_group
            self.vp_data_tokenized[group_no] = vp_data_tokenized_in_group
            self.vp_yn_idx[group_no] = vp_yn_idx_in_group
        self.ready_to_predict = True
        self.load_doc2vec(root, user, project, data_type)        
    
    def predict(self, x):
        try_cnt = 0                
        while self.ready_to_predict == False:
            if self.error:
                return '', '', ''
            if try_cnt > self.retry_limit:
                return '', '', ''
            time.sleep(self.SLEEP_TIME)
            try_cnt += 1
        
        similar_sample_res = []
        max_prob_res = 0
        g_nouns = {}
        for group_no in self.vp_data_nouns:
            nnouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(x.replace('\n', '^')), self.vp_yn[group_no])
            g_nouns[group_no] = []
            for i in range(len(nnouns)):
                if self.voca_weight[group_no].get(nnouns[i], None) != None:
                    g_nouns[group_no].append(nnouns[i])
        for group_no in self.vp_data_nouns:
            nouns = g_nouns[group_no]
            sample_nouns = []
            sample_with_tag = []
            sample_tokenized = []
            x_vp_yn_cnt = 0            
            for n in nouns:
                if self.vp_yn.get(group_no, '') != '':
                    if self.vp_yn[group_no].get(n, '') == 'Y':
                        x_vp_yn_cnt += 1
            vp_yn_cnt = {}
            for i in range(len(nouns)):
                vp_yn_data_idx_arr = self.vp_yn_idx[group_no].get(nouns[i], '')
                if vp_yn_data_idx_arr != '':
                    for vp_yn_data_idx in vp_yn_data_idx_arr:
                        if vp_yn_cnt.get(vp_yn_data_idx, '') == '':
                            vp_yn_cnt[vp_yn_data_idx] = 1
                        else:
                            vp_yn_cnt[vp_yn_data_idx] += 1
            
            for key, value in vp_yn_cnt.items():
                if value >= 2:
                    sample_nouns.append(self.vp_data_nouns[group_no][key])                
                    sample_with_tag.append(self.vp_data_with_tag[group_no][key])
                    sample_tokenized.append(self.vp_data_tokenized[group_no][key])
            x = nouns
            if len(sample_with_tag) > 0:
                max_prob, similar_sample = self.get_ita_algo_score(x, sample_tokenized, sample_nouns, group_no)
                if max_prob > 60:
                    doc2vec_score = self.get_doc2vec_score(nouns, group_no)
                    doc2vec_prob = doc2vec_score[0][1]                    
                    max_prob = min(max(round(max_prob * 0.6 + doc2vec_prob * 0.4), max_prob), 100)
                if len(similar_sample) == 0:
                    similar_sample = [['Not Found', 0]]
                max_prob_res = max(max_prob, max_prob_res)
                for ss in similar_sample:
                    similar_sample_res.append(ss)
            else:
                similar_sample = ['Not Found', 0]            
                similar_sample_res.append(similar_sample)
                
        similar_sample_res = [sorted(similar_sample_res, key=lambda item: item[1], reverse=True)[:5]]
        return max_prob_res, similar_sample_res, tokenized
        
    def get_ita_algo_score(self, x, sample, nouns, group_no):
        similar_sample = []            
        max_prob = 0
        for i in range(len(nouns)):
            d = {}
            prob = round(ita_algo.get_prob(x, nouns[i], self.voca_weight[group_no]) * 100)
            if prob == 0 or prob == 100:
                continue 
            d['res'] = [sample[i], prob, nouns[i]]
            max_prob = max(prob, max_prob)
            similar_sample.append(d)
        
        similar_sample = sorted(similar_sample, key=lambda item: item['res'][1], reverse=True)
        
        res = []
        for i in range(min(len(similar_sample), 5)):
            tokenized_text = similar_sample[i]['res'][0]
            prob = similar_sample[i]['res'][1]
            nouns = similar_sample[i]['res'][2]
            res.append([self.get_part_of_tokenized_text(tokenized_text, nouns), prob])

        return max_prob, res

    def get_part_of_tokenized_text(self, tokenized_text, nouns):
        last_word = nouns[len(nouns) - 1]
        last_word_cnt = 0
        for n in nouns:
            if n == last_word:
                last_word_cnt += 1
        tt_word_cnt = 0
        res = ''
        for tt in tokenized_text:
            res += tt + " "
            if tt == last_word:
                tt_word_cnt += 1
            if tt_word_cnt == last_word_cnt:
                return res
        
        return res

    def load_doc2vec(self, root, user, project, data_type):
        model_path = os.path.join(root, user, project, data_type, 'working_dir')
        model_list = os.listdir(model_path)
        for each_model in model_list:
            group_no = each_model.split('.')[2]
            self.doc2vec_model[group_no] = g.Doc2Vec.load(os.path.join(model_path, each_model))
                        
    def get_doc2vec_score(self, x, group_no):
        similar_sample = []
        for key in self.doc2vec_model:            
            if key == group_no:
                new_vector = self.doc2vec_model[key].infer_vector(x)
                res = self.doc2vec_model[key].docvecs.most_similar([new_vector])
                for i in range(len(res)):
                    prob = round(res[i][1] * 100)
                    similar_sample.append([res[i][0], prob])
                return similar_sample
        return None        

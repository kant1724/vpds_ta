import os
import time
from detection.extension import tokenization
from detection.models.b_jaro_winkler.engine import jaro_wrinkler

class runner():
    vp_yn = {}
    voca_entity = {}
    vp_data = []
    vp_data_nouns = []
    vp_data_with_tag = []
    vp_data_tokenized = []
    vp_yn_idx = {}
    ready_to_predict = False
    error = False
    SLEEP_TIME = 0.5
    retry_limit = 200
    
    def init(self, root, user, project, data_type):
        self.vp_yn = {}
        self.voca_weight = {}
        self.voca_entity = {}
        self.vp_data_nouns = []
        self.vp_data_with_tag = []
        self.vp_data_tokenized = []
        self.vp_yn_idx = {}
        self.ready_to_predict = False
        self.error = False
        path = os.path.join(root, user, project, data_type, 'vp_data_file')
        self.error = False
        with open(os.path.join(path, 'voca_data.txt'), 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                arr = line.split('^')
                self.voca_entity[arr[0]] = arr[1]
                self.vp_yn[arr[0]] = arr[2]
                self.voca_weight[arr[0]] = arr[3]
        
        with open(os.path.join(path, 'vp_data_nouns.txt'), 'r', encoding='utf8') as f1:
            with open(os.path.join(path, 'tokenized_vp_data.txt'), 'r', encoding='utf8') as f2:
                vp_data_idx = 0
                lines = f1.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    nouns = line.split(' ')
                    for i in range(len(nouns)):
                        if self.vp_yn.get(nouns[i], '') == 'Y':
                            if self.vp_yn_idx.get(nouns[i], '') != '':
                                if vp_data_idx not in self.vp_yn_idx[nouns[i]]:
                                    self.vp_yn_idx[nouns[i]].append(vp_data_idx)
                            else:
                                self.vp_yn_idx[nouns[i]] = [vp_data_idx]
                    self.vp_data_nouns.append(nouns)
                    self.vp_data_with_tag.append(tokenization.tagging_words(nouns, self.voca_entity))
                    vp_data_idx += 1
                lines = f2.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    tokenized = line.split(' ')
                    self.vp_data_tokenized.append(tokenized)
            
        self.ready_to_predict = True
    
    def predict(self, x, min_vp_voca_same_rate, vp_threshold, less_threshold_decrease_point):
        try_cnt = 0                
        while self.ready_to_predict == False:
            if self.error:
                return '', '', ''
            if try_cnt > self.retry_limit:
                return '', '', ''
            time.sleep(self.SLEEP_TIME)
            try_cnt += 1
            
        sample_nouns = []
        sample_with_tag = []
        sample_tokenized = []
        nouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(x.replace('\n', '^')), self.vp_yn)
        x_vp_yn_cnt = 0
        for n in nouns:
            if self.vp_yn.get(n, '') == 'Y':
                x_vp_yn_cnt += 1
        vp_yn_cnt = {}
        print(nouns)
        for i in range(len(nouns)):
            vp_yn_data_idx_arr = self.vp_yn_idx.get(nouns[i], '')
            print(vp_yn_data_idx_arr)
            if vp_yn_data_idx_arr != '':
                for vp_yn_data_idx in vp_yn_data_idx_arr:
                    if vp_yn_cnt.get(vp_yn_data_idx, '') == '':
                        vp_yn_cnt[vp_yn_data_idx] = 1
                    else:
                        vp_yn_cnt[vp_yn_data_idx] += 1
        for key, value in vp_yn_cnt.items():
            if value >= max(int(x_vp_yn_cnt * float(min_vp_voca_same_rate)), 2):
                sample_nouns.append(self.vp_data_nouns[key])                
                sample_with_tag.append(self.vp_data_with_tag[key])
                sample_tokenized.append(self.vp_data_tokenized[key])
        x = nouns
        print(sample_with_tag)
        if len(sample_with_tag) > 0:
            max_prob, similar_sample = self.get_jaro_winkler_score(x, sample_tokenized, sample_nouns, vp_threshold, less_threshold_decrease_point)
            if len(similar_sample) == 0:
                similar_sample = [['Not Found', 0]]
        else:
            max_prob, similar_sample = 0, [['Not Found', 0]]
            
        return max_prob, similar_sample, tokenized
        
    def get_jaro_winkler_score(self, x, sample, nouns, vp_threshold, less_threshold_decrease_point):
        similar_sample = []            
        max_prob = 0        
        for i in range(len(nouns)):
            if len(x) > len(nouns[i]):
                continue
            sample_len = min(len(nouns[i]), max(len(x), 200))
            d = {}
            prob = round(jaro_wrinkler.new_jaro_wrinkler(x, nouns[i][:sample_len], self.voca_weight) * 100)
            print("prob: " + str(prob))
            if prob == 0:
                continue
            if prob < int(vp_threshold):
                prob = max(prob - int(less_threshold_decrease_point), 0)
            d['res'] = [sample[i], prob, nouns[i], sample_len]
            max_prob = max(prob, max_prob)
            similar_sample.append(d)
        
        similar_sample = sorted(similar_sample, key=lambda item: item['res'][1], reverse=True)
        
        res = []
        for i in range(min(len(similar_sample), 5)):
            tokenized_text = similar_sample[i]['res'][0]
            prob = similar_sample[i]['res'][1]
            nouns = similar_sample[i]['res'][2]
            sample_len = similar_sample[i]['res'][3]
            res.append([self.get_part_of_tokenized_text(tokenized_text, nouns[:sample_len]), prob])

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

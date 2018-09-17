import os
import gensim.models as g
from detection.extension import tokenization

class runner():
    tokenized_vp_data = {}
    voca_entity = {}
    vp_yn = {}
    model = {}
    def init(self, root, user, project, data_type):
        self.tokenized_vp_data = {}
        self.voca_entity = {}
        self.vp_yn = {}
        self.model = {}
        voca_path = os.path.join(root, user, project, data_type, 'vp_data_file', 'voca_data.txt')
        with open(os.path.join(voca_path), 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                arr = line.split('^')
                self.voca_entity[arr[0]] = arr[1]
                self.vp_yn[arr[0]] = arr[2]
        model_path = os.path.join(root, user, project, data_type, 'working_dir')
        model_list = os.listdir(model_path)
        for each_model in model_list:
            group_no = each_model.split('.')[2]
            self.model[group_no] = g.Doc2Vec.load(os.path.join(model_path, each_model))
            vp_data_path = os.path.join(root, user, project, data_type, 'vp_data_file', group_no, 'tokenized_vp_data.txt')
            d = []
            with open(vp_data_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    d.append(line)
            self.tokenized_vp_data[group_no] = d
        
    def run(self, x):
        nouns, tokenized = tokenization.extract_vp_word_in_pos(tokenization.pos(x.replace('\n', '^')), self.vp_yn)
        max_prob = 0
        similar_sample = []
        for key in self.model:            
            new_vector = self.model[key].infer_vector(nouns)
            res = self.model[key].docvecs.most_similar([new_vector])
            for i in range(5):
                prob = round(res[i][1] * 100)
                tokenized_vp_text = self.tokenized_vp_data[key][res[i][0]]
                if tokenized == tokenized_vp_text.split(" "):
                    continue
                similar_sample.append([tokenized_vp_text, prob])
                max_prob = max(max_prob, prob)
        similar_sample = sorted(similar_sample, key=lambda x: x[1], reverse=True)[:5]
        
        return max_prob, similar_sample, tokenized

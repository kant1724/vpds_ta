from detection.models.c_doc2vec import properties as p
import os
import shutil
import gensim.models as g
import logging

class trainer():
    vector_size = 300
    window_size = 15
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 0
    worker_count = 1
    pretrained_emb = None
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    end_yn = 'N'
    
    def train(self, user, project, data_type, end_step):
        self.train_epoch = int(end_step)
        train_files_dir = os.path.join(p.get_root(), user, project, data_type, 'vp_data_file') 
        dir_list = os.listdir(train_files_dir)
        working_dir = os.path.join(p.get_root(), user, project, data_type, 'working_dir')
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir)
        for i in range(len(dir_list)):
            group_dir = os.path.join(train_files_dir, dir_list[i])
            if os.path.isdir(group_dir) == False:
                continue
            train_files_path = os.path.join(group_dir, 'vp_data_nouns.txt')
            save_path = os.path.join(working_dir, 'model.bin.' + str(dir_list[i]))
            docs = g.doc2vec.TaggedLineDocument(train_files_path)
            
            model = g.Doc2Vec(docs, size=self.vector_size, window=self.window_size,
                              min_count=self.min_count, sample=self.sampling_threshold,
                              workers=self.worker_count, hs=0, dm=self.dm, negative=self.negative_size,
                              dbow_words=1, dm_concat=1, iter=self.train_epoch)
            
            model.save(save_path)
            self.end_yn = 'Y'

    def is_end(self):
        return self.end_yn

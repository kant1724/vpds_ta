import re
import os
from detection.models.a_cnn.file_processor import training_file_manager

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

exception = ['?', ' ', '.', '"', ','] 

def representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def char_tokenizer(sentence):
    words = []
    i = 0
    while i < len(sentence):
        if representsInt(sentence[i]):
            num = ""
            while i < len(sentence) and representsInt(sentence[i]):
                num += str(sentence[i])
                i += 1
            words.append(num.replace('\n', ''))
        else:
            if sentence[i] in exception:
                i += 1
                continue
            words.append(sentence[i].replace('\n', ''))
            i += 1
    return words

def word_tokenizer(sentence):
    return sentence.replace("\n", "").split(" ")

def create_vocabulary(train_arr, max_vocabulary_size, flag, language, root, user, project, data_type, slice_type):
    vocab = {}
    for line in train_arr:
        if flag == 'enc':
            if language == 'eng':
                tokens = word_tokenizer(line)
            elif language == 'kor':
                tokens = word_tokenizer(line)
        else:
            tokens = line.replace('\n', '').split(" ")
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

    create_vocabulary_file(flag, vocab_list, root, user, project, data_type, slice_type)
    
def create_vocabulary_file(flag, vocab_list, root, user, project, data_type, slice_type):
    file_name = ''
    if flag == 'enc':
        file_name = 'vocab.enc'
    else:
        file_name = 'vocab.dec'
    path = os.path.join(root, user, project, data_type, slice_type, 'training_files')
    with open(os.path.join(path, file_name), 'w', encoding='utf8') as fw:
        for w in vocab_list:
            fw.write(w + "\n")

def initialize_vocabulary(flag, root, user, project, data_type, slice_type):
    file_name = ''
    if flag == 'enc':
        file_name = 'vocab.enc'
    else:
        file_name = 'vocab.dec'
    path = os.path.join(root, user, project, data_type, slice_type, 'training_files')
    all_vocab = {}
    with open(os.path.join(path, file_name), 'r', encoding='utf8') as fw:
        cnt = 1
        for line in fw.readlines():
            line = line.replace('\n', '')
            if flag == 'enc':
                all_vocab[line] = cnt
            else:
                all_vocab[cnt] = line
            cnt += 1
    
    return all_vocab

def sentence_to_token_ids(sentence, vocabulary, language):
    if language == 'eng':
        words = word_tokenizer(sentence)
    elif language == 'kor':
        words = word_tokenizer(sentence)
    
    return [str(vocabulary.get(w, UNK_ID)) for w in words]

def data_to_token_ids(train_arr, flag, language, root, user, project, data_type, slice_type):
    vocab = initialize_vocabulary(flag, root, user, project, data_type, slice_type)
    rev_vocab = {}
    token_arr = []
    for k, v in vocab.items():
        rev_vocab[v] = k
    for line in train_arr:
        if flag == 'enc':
            token_ids = sentence_to_token_ids(line, vocab, language)
            token_arr.append(" ".join(token_ids))
        else:
            ll = line.replace('\n', '').split(" ")
            token_ids = []
            for l in ll:
                token_ids.append(str(rev_vocab[l]))
            token_arr.append(" ".join(token_ids))
    return token_arr
    
def prepare_custom_data(root, user, project, data_type, slice_type, enc_vocabulary_size, dec_vocabulary_size, language):
    train_enc, train_dec = training_file_manager.get_training_data(root, user, project, data_type, slice_type)
    
    create_vocabulary(train_enc, enc_vocabulary_size, 'enc', language, root, user, project, data_type, slice_type)
    create_vocabulary(train_dec, dec_vocabulary_size, 'dec', language, root, user, project, data_type, slice_type)

    train_enc_ids = data_to_token_ids(train_enc, 'enc', language, root, user, project, data_type, slice_type)
    train_dec_ids = data_to_token_ids(train_dec, 'dec', language, root, user, project, data_type, slice_type)

    return train_enc_ids, train_dec_ids, train_enc, train_dec

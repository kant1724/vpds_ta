from detection.models.c_doc2vec.engine import run
from detection.models.a_cnn import properties as p

runner = None
vocab = []

def get_answer(root, user, project, data_type, x):
    global runner, vocab
    if runner == None:
        runner = run.runner()
        runner.init(root, user, project, data_type)
    predict_result, similar_sample, tokenized = runner.run(x)
    
    return predict_result, similar_sample, tokenized

def remove_runner():
    global runner
    runner = None

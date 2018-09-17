from detection.models.a_cnn.engine import run
from detection.models.a_cnn.file_processor import training_file_creator
from detection.models.a_cnn import properties as p

runner = None
vocab = []

def get_answer(user, project, data_type, slice_type, x):
    global runner, vocab
    if runner == None:
        runner = run.runner()
        runner.init(user, project, data_type, slice_type)
    root = p.get_root()
    if len(vocab) == 0:
        vocab = training_file_creator.initialize_vocabulary('enc', root, user, project, data_type, slice_type)
    train_enc_ids = [" ".join(training_file_creator.sentence_to_token_ids(x, vocab, 'kor'))]
    predict_result = runner.predict(train_enc_ids)

    return predict_result

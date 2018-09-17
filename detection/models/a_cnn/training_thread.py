from threading import Thread
from detection.models.a_cnn.engine import train
trainer_thread = []

def start_training_thread(user, project, data_type, slice_type, end_step, train_enc_ids, train_dec):
    trainer = train.trainer()
    thread = Thread(target = trainer.train, args = (user, project, data_type, slice_type, end_step, train_enc_ids, train_dec))
    thread.start()
    trainer_thread.append({"user" : user, "project" : project, "data_type" : data_type, "trainer" : trainer})

def stop_training_thread(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            trainer = trainer_thread[i]['trainer']
            trainer.stop()
            trainer_thread.remove(trainer_thread[i])
            break
    
def get_trainer_thread(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            return trainer_thread[i]['trainer']
    return None

def training_test(user, project, data_type, token_ids):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            trainer = trainer_thread[i]['trainer']
            res = trainer.training_test(token_ids)
            return res

def get_trainer_object(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            return trainer_thread[i]['trainer']

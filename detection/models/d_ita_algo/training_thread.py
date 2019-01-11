from threading import Thread
from detection.models.d_ita_algo.engine import train
trainer_thread = []

def start_training_thread(user, project, data_type, end_step):
    trainer = train.trainer()
    thread = Thread(target = trainer.train, args = (user, project, data_type, end_step))
    thread.start()
    trainer_thread.append({"user" : user, "project" : project, "data_type" : data_type, "trainer" : trainer})

def remove_training_thread(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            trainer_thread.remove(trainer_thread[i])
            break

def get_trainer_thread(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            return trainer_thread[i]['trainer']
    return None

def get_trainer_object(user, project, data_type):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project and trainer_thread[i]['data_type'] == data_type:
            return trainer_thread[i]['trainer']

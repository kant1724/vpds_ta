from detection.models.d_ita_algo.engine import run

runner_thread = []
bulk_of_answer_thread = []

def remove_runner(user, project, data_type):
    global runner_thread
    for i in range(len(runner_thread)):
        if runner_thread[i]['user'] == user and runner_thread[i]['project'] == project and runner_thread[i]['data_type'] == data_type:            
            runner_thread.remove(runner_thread[i])
    
def get_answer(root, user, project, data_type, x):
    global runner_thread
    runner = None
    for i in range(len(runner_thread)):
        if runner_thread[i]['user'] == user and runner_thread[i]['project'] == project and runner_thread[i]['data_type'] == data_type:
            runner = runner_thread[i]['runner']
            break
    if runner == None:
        runner = run.runner()
        runner.init(root, user, project, data_type)
        runner_thread.append({"user" : user, "project" : project, "data_type" : data_type, "runner" : runner})
    predict_result, similar_sample, tokenized = runner.predict(x)
    
    return predict_result, similar_sample, tokenized

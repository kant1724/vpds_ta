from django.http import JsonResponse
from detection.models.a_cnn import worker as cnn_worker
from detection.models.b_jaro_winkler import worker as jaro_winkler_worker
from detection.models.c_doc2vec import worker as doc2vec_worker
from detection.models.d_ita_algo import worker as ita_algo_worker
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def get_answer_from_cnn(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    slice_type = request.POST.get('slice_type', '')
    x = request.POST.get('x', '')
    similar_sample = ''
    cnn_training_target = request.POST.get('cnn_training_target', '')
    predict_result= cnn_worker.get_answer(user, project, data_type, slice_type, x, cnn_training_target)
    
    return JsonResponse({"predict_result" : predict_result, "similar_sample" : similar_sample}) 

@csrf_exempt
def get_answer_from_jw(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    x = request.POST.get('x', '')
    min_vp_voca_same_rate = request.POST.get('min_vp_voca_same_rate', '')
    vp_threshold = request.POST.get('vp_threshold', '')
    less_threshold_decrease_point = request.POST.get('less_threshold_decrease_point', '')
    predict_result, similar_sample, tokenized = jaro_winkler_worker.get_answer(user, project, data_type, x
                                                                             , min_vp_voca_same_rate, vp_threshold, less_threshold_decrease_point)
    
    return JsonResponse({"predict_result" : predict_result, "similar_sample" : similar_sample, "tokenized" : tokenized})

@csrf_exempt
def get_answer_from_doc2vec(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    x = request.POST.get('x', '')
    predict_result, similar_sample, tokenized = doc2vec_worker.get_answer(user, project, data_type, x)
    
    return JsonResponse({"predict_result" : predict_result, "similar_sample" : similar_sample, "tokenized" : tokenized})

@csrf_exempt
def get_answer_from_ita_algo(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    x = request.POST.get('x', '')
    predict_result, similar_sample, tokenized, max_prob_group_no = ita_algo_worker.get_answer(user, project, data_type, x)
    
    return JsonResponse({"predict_result" : predict_result, "similar_sample" : similar_sample, "tokenized" : tokenized, 'max_prob_group_no' : max_prob_group_no})

@csrf_exempt
def start_training(request):
    user, project, data_type, end_step, language = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', ''), request.POST.get('end_step', ''), request.POST.get('language', '')
    x, y = eval(request.POST.get('x', '')), eval(request.POST.get('y', ''))
    voca_list = eval(request.POST.get('voca_list', ''))
    model_type = request.POST.get('model_type', '')
    slice_type = request.POST.get('slice_type', '')
    if model_type == '1':
        cnn_training_target = request.POST.get('cnn_training_target', '')
        cnn_worker.upload_training_data(user, project, data_type, slice_type, x, y, cnn_training_target)
        cnn_worker.start_training(user, project, data_type, slice_type, end_step, language)
    elif model_type == '3':
        doc2vec_worker.upload_training_data(user, project, data_type, x, voca_list)
        doc2vec_worker.start_training(user, project, data_type, end_step)
    elif model_type == '4':
        ita_algo_worker.upload_vp_data(user, project, data_type, x, voca_list)
        ita_algo_worker.start_training(user, project, data_type, end_step)
    
    return JsonResponse({})

@csrf_exempt
def stop_training(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    model_type = request.POST.get('model_type', '')
    if model_type == '1':
        cnn_worker.stop_training(user, project, data_type)
        
    return JsonResponse({})

@csrf_exempt
def get_training_info(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    model_type = request.POST.get('model_type', '')
    if model_type == '1':
        training_info, end_step = cnn_worker.get_training_info(user, project, data_type)
    elif model_type == '3':
        training_info, end_step = doc2vec_worker.get_training_info(user, project, data_type)
    elif model_type == '4':
        training_info, end_step = ita_algo_worker.get_training_info(user, project, data_type)
    
    return JsonResponse({"training_info" : training_info, "end_step" : end_step})

@csrf_exempt
def is_training(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    model_type = request.POST.get('model_type', '')
    if model_type == '1':
        is_training_yn = cnn_worker.is_training(user, project, data_type)
    elif model_type == '3':
        is_training_yn = doc2vec_worker.is_training(user, project, data_type)
    elif model_type == '4':
        is_training_yn = ita_algo_worker.is_training(user, project, data_type)
    else:
        is_training_yn = 'N' 
    
    return JsonResponse({"is_training_yn" : is_training_yn})

@csrf_exempt
def delete_ckpt(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    model_type = request.POST.get('model_type', '')
    slice_type = request.POST.get('slice_type', '')
    if model_type == '1':
        cnn_worker.delete_ckpt(user, project, data_type, slice_type)
    
    return JsonResponse({})

@csrf_exempt
def get_probability(request):
    user, project, data_type = 'vpds', 'vpds', 'vp_data'
    req = json.loads(request.body)
    x = req.get('contents', '')
    min_vp_voca_same_rate = 0
    vp_threshold = 50
    less_threshold_decrease_point = 10
    predict_result, _, _, _ = ita_algo_worker.get_answer(user, project, data_type, x)
        
    return JsonResponse({"reply" : float(predict_result) / 100})

@csrf_exempt
def upload_vp_data(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    vp_data = eval(request.POST.get('vp_data', ''))
    voca_list = eval(request.POST.get('voca_list', ''))
    model_type = request.POST.get('model_type', '')
    uploading_config = request.POST.get('uploading_config', '')
    key = request.POST.get('key', '')
    if key != 'jw@1234':
        print("invalid key!!!!!!!!!!!!")
        return JsonResponse({})
    if model_type == '2':
        jaro_winkler_worker.upload_vp_data(user, project, data_type, vp_data, voca_list, uploading_config)
    
    return JsonResponse({})

@csrf_exempt
def get_uploading_config(request):
    user, project, data_type = request.POST.get('user', ''), request.POST.get('project', ''), request.POST.get('data_type', '')
    model_type = request.POST.get('model_type', '')
    if model_type == '2':
        uploading_config = jaro_winkler_worker.get_uploading_config(user, project, data_type)
        
    return JsonResponse({"uploading_config" : uploading_config})

@csrf_exempt
def check_online(request):
    return JsonResponse({})

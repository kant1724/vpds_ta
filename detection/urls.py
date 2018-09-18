"""vpds URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from detection import views

urlpatterns = [
    path('get_answer_from_cnn', views.get_answer_from_cnn, name='get_answer_from_cnn'),
    path('get_answer_from_jw', views.get_answer_from_jw, name='get_answer_from_jw'),
    path('get_answer_from_doc2vec', views.get_answer_from_doc2vec, name='get_answer_from_doc2vec'),
    path('get_probability', views.get_probability, name='get_probability'),
    path('start_training', views.start_training, name='start_training'),
    path('stop_training', views.stop_training, name='stop_training'),
    path('get_training_info', views.get_training_info, name='get_training_info'),
    path('is_training', views.is_training, name='is_training'),
    path('delete_ckpt', views.delete_ckpt, name='delete_ckpt'),
    path('upload_vp_data', views.upload_vp_data, name='upload_vp_data'),
    path('get_uploading_config', views.get_uploading_config, name='get_uploading_config'),
    path('check_online', views.check_online, name='check_online')
]

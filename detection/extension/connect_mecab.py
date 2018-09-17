import requests
import json
import properties

mecab_ip = properties.get_mecab_ip()

def pos(text):
    url = 'http://' + mecab_ip + '/pos'
    response = requests.post(url, data={"text" : text})
    return json.loads(response.content.decode('utf8'))

def morphs(text):
    url = 'http://' + mecab_ip + '/morphs'
    response = requests.post(url, data={"text" : text})
    return json.loads(response.content.decode('utf8'))

def nouns(text):
    url = 'http://' + mecab_ip + '/nouns'
    response = requests.post(url, data={"text" : text})
    return json.loads(response.content.decode('utf8'))

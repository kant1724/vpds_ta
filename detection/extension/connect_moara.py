import requests
import json
import properties

moara_ip = properties.get_moara_ip()
                                   
def pos(text):
    url = 'http://' + moara_ip + '/word/extract'
    data = {}
    data['langCode'] = "KO"
    data['sentenceValue'] = text
    response = requests.post(url, data=json.dumps(data))
    json_arr = json.loads(response.content.decode('utf8'))
    pos = []
    for i in range(len(json_arr)):
        pos.append((json_arr[i]['syllable'], json_arr[i]['wordClassDetail']))

    return pos

def morphs(text):
    url = 'http://' + moara_ip + '/word/extract'
    data = {}
    data['langCode'] = "KO"
    data['sentenceValue'] = text
    response = requests.post(url, data=json.dumps(data))
    json_arr = json.loads(response.content.decode('utf8'))
    morphs = []
    for i in range(len(json_arr)):
        morphs.append(json_arr[i]['syllable'])

    return morphs

def nouns(text):
    url = 'http://' + moara_ip + '/word/extract'
    data = {}
    data['langCode'] = "KO"
    data['sentenceValue'] = text
    response = requests.post(url, data=json.dumps(data))
    json_arr = json.loads(response.content.decode('utf8'))
    nouns = []
    for i in range(len(json_arr)):
        if json_arr[i]['wordClass'] == 'NOUN':
            nouns.append(json_arr[i]['syllable'])
    
    return nouns

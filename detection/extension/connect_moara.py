import requests
import json
import properties

moara_ip = properties.get_moara_ip()
                                   
def pos(text):
    url = moara_ip + '/document/word/extract'
    data = {}
    data['contents'] = text
    data['isCompound'] = False
    response = requests.post(url, data=json.dumps(data), verify=False)
    json_arr = json.loads(response.content.decode('utf8'))
    pos = []
    for i in range(len(json_arr)):
        if json_arr[i]['extractType'] == 'CHANGE':
            continue
        pos.append((json_arr[i]['syllable'], json_arr[i]['wordClassDetail']))

    return pos

def morphs(text):
    url = moara_ip + '/document/word/extract'
    data = {}
    data['contents'] = text
    data['isCompound'] = False
    response = requests.post(url, data=json.dumps(data), verify=False)
    json_arr = json.loads(response.content.decode('utf8'))
    morphs = []
    for i in range(len(json_arr)):
        if json_arr[i]['extractType'] == 'CHANGE':
            continue
        morphs.append(json_arr[i]['syllable'])

    return morphs

def nouns(text):
    url = moara_ip + '/document/word/extract'
    data = {}
    data['contents'] = text
    data['isCompound'] = False
    response = requests.post(url, data=json.dumps(data), verify=False)
    json_arr = json.loads(response.content.decode('utf8'))
    nouns = []
    for i in range(len(json_arr)):
        if json_arr[i]['extractType'] == 'CHANGE':
            continue
        if json_arr[i]['wordClass'] == 'NOUN':
            nouns.append(json_arr[i]['syllable'])
    
    return nouns

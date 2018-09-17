with open('./server_ip', encoding="utf8") as f:
    lines = f.readlines()
    MECAB_IP = lines[0].split("=")[1].replace('\n', '')
    MOARA_IP = lines[1].split("=")[1].replace('\n', '')
    
def get_mecab_ip():
    return MECAB_IP

def get_moara_ip():
    return MOARA_IP

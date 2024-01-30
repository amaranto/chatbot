import re 

def cleaning(s, max_length: int = 0):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace(",","")
    s = s.replace('"',"")
    return s if max_length <= 0 else s[:max_length]


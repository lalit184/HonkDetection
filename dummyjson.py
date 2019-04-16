a={}
a["test1.wav"]="test1.txt"
import json

with open('dum.json', 'w') as fp:
    json.dump(a, fp)
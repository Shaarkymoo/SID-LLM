import json
import csv

filename = r"D:\college\Sem5\dsn2099\DecodingTrust-main\data\toxicity\user_prompts\toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl"

data = []
with open(filename,'r',encoding='utf8') as box:
    for a in box:
        data.append(json.loads(a))

#print(data)
prompts=[]
for a in data:
    text1 = str(a["prompt"]["text"])
    #print(text1)
    try:
        text2 = str(a["prompt"]["continuation"]["text"])
        #print(text2)
    except Exception as e:
        text2 = False

    prompts.append(text1 + (text2 if text2 else ""))

csvfile = r"D:\college\Sem5\dsn2099\DecodingTrust-main\data\toxicity\user_prompts\filtered toxic prompts from gpt3.csv"

#print(prompts)

with open(csvfile,'w',encoding='utf8') as box:
    writer = csv.writer(box)
    writer.writerows([[prompt] for prompt in prompts if prompt!="\n"])


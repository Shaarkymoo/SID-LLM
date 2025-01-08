import json
import csv

filename = (r"D:\college\Sem5\dsn2099\beavertails-main\train300.jsonl")
data=[]
with open(filename,'r',encoding='utf8') as box:
    for a in box:
        data.append(json.loads(a))

labels = []
prompts=[]
bene_count = 0
for a in data:
    text1 = str(a["prompt"])
    count = 0
    if list(a["category"].values()).count(True)>1:
        continue
    else:
        if list(a["category"].values()).count(True)==0:
            key = "benevolent"
            bene_count+=1
        else:
            trueindex = list(a["category"].values()).index(True)
            key = str(list(a["category"].keys())[trueindex])

    labels.append(key)
    prompts.append(text1)

csvfile = r"D:\college\Sem5\dsn2099\beavertails-main\train300.csv"

print(len(prompts))
print(len(labels))
print(bene_count)
truelabels = []
for a in labels:
    if a not in truelabels:
        truelabels.append(a)

print(truelabels)
rows = []

for a in range(len(prompts)):
    rows.append([labels[a],prompts[a]])

with open(csvfile,'w',encoding='utf8') as box:
    writer = csv.writer(box)
    writer.writerows(rows)
#print(data)
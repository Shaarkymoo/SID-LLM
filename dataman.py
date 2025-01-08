import csv

filename = r"D:\college\Sem5\dsn2099\beavertails-main\train300.csv"

fields = []

train_prompts = []
train_labels = []
rows = []

with open(filename,'r',encoding = 'utf8') as box:
    csvreader = csv.reader(box)
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)
        if len(row)==0:
            continue

        if row[0] not in train_labels:
            train_labels.append(row[0])
            
        train_prompts.append(row[1])
    
    print("Total no of rows: %d" % (len(train_prompts)))

#print('Field names are:' + ', '.join(field for field in fields))

#for row in rows:
    #print(row)

print(train_labels)

# for a in range(len(train_prompts)):
#     print(train_prompts[a],train_labels[a],sep=', ')



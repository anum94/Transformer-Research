from ds.supported import load_dataset


folder_names = ['falcon7b-arxiv',
'falcon7b-pubmed',
'bart-arxiv',
'bart-arxiv-1024-19062023153234',
'bart-govreport',
'bart-pubmed',
'bigbirdpegasus-arxiv',
'bigbirdpegasus-arxiv-4096-15062023193353',
'bigbirdpegasus-govreport',
'bigbirdpegasus-govreport-4096-15062023190336',
'bigbirdpegasus-pubmed',
'bigbirdpegasus-pubmed-4096-17062023183256',
'gpt-3.5-turbo-0613-arxiv',
'gpt-3.5-turbo-0613-govreport',
'gpt-3.5-turbo-0613-pubmed',
'gpt-3.5-turbo-16k-0613-arxiv',
'gpt-3.5-turbo-16k-0613-govreport',
'gpt-3.5-turbo-16k-0613-pubmed',
'pegasusx-arxiv-4096-12062023155841',
'pegasusx-arxiv-8192-17062023185926',
'pegasusx-govreport-4096-15062023183547',
'pegasusx-govreport-8192-17062023195045',
'pegasusx-pubmed-4096-12062023145646',
'pegasusx-pubmed-8192-17062023204832',
'falcon40b-govreport',
                ]


import os.path, glob
summaries_arxiv = dict()
summaries_pubmed = dict()
summaries_govreport = dict()


for fname in folder_names:

    path = os.path.join('results', fname)
    print(path)
    fname_prediction = glob.glob(path + '/*.out')
    print(fname_prediction)
    with open(fname_prediction[0]) as f:
        lines = f.readlines()
        if len(lines) == 1:
            lines = lines[0]
            lines = eval (lines)

        if 'arxiv' in fname:
            summaries_arxiv[fname] = lines
        elif 'pubmed' in fname:
            summaries_pubmed[fname] = lines
        else:
            summaries_govreport[fname] = lines


print(summaries_arxiv.keys())
print(summaries_pubmed.keys())
print(summaries_govreport.keys())

import random
no_samples_arxiv = 6440
no_samples_govreport = 973
no_samples_pubmed = 6658



r = no_samples_arxiv-1
idx=random.randint(0,r)
print ("Index: ", idx)

# optional to print the article

# load/process ds
dataset = load_dataset(
    dataset="arxiv",
    preview=False,
    samples="max",
    min_input_size=0,
)
arxiv_dtest = dataset.get_split("test")
sample_article = arxiv_dtest['text'][idx]
sample_summary = arxiv_dtest['summary'][idx]
print ("Printing the article: \n")
print (sample_article)

print ("Summary: \n")
print (sample_summary)

for key, value in summaries_arxiv.items():
    print (key)
    try:
        print(value[idx])
    except:
        print ("{Summary missing}")



r = no_samples_pubmed-1
idx=random.randint(0,r)
print ("Index: ", idx)


# optional to print the article

# load/process ds
dataset = load_dataset(
    dataset="pubmed",
    preview=False,
    samples="max",
    min_input_size=0,
)
pubmed_dtest = dataset.get_split("test")
sample_article = pubmed_dtest['text'][idx]
print ("Printing the article: \n")
print (sample_article)

for key, value in summaries_pubmed.items():
    print (key)
    try:
        print(value[idx])
    except:
        print ("{Summary missing}")

r = no_samples_govreport-1
idx=random.randint(0,r)
print ("Index: ", idx)
# optional to print the article

# load/process ds
dataset = load_dataset(
    dataset="govreport",
    preview=False,
    samples="max",
    min_input_size=0,
)
govreport_dtest = dataset.get_split("test")
sample_article = govreport_dtest['text'][idx]
print ("Printing the article: \n")
print (sample_article)

for key, value in summaries_govreport.items():
    print (key)
    try:
        print(value[idx])
    except:
        print ("{Summary missing}", len(value))
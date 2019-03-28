# %%
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize

# %% import datasets
# cwd = os.getcwd()
traindata = pd.read_csv(filepath_or_buffer="./fortunecookiedata/traindata.txt",header=None,names=['traindata'])
trainlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/trainlabels.txt',names=['trainlabels'])

stopwords = pd.read_csv(filepath_or_buffer="./fortunecookiedata/stoplist.txt",header=None,names=['stopwords'])

testdata = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testdata.txt',header=None,names=['testdata'])
testlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testlabels.txt',header=None,names=['testlabels'])


'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
# %% ----------- pre-processing -----------
# traindata.columns = ['traindata']
traindata.head()
# %% --------------Tokenize traindata-----------
x = traindata['traindata'].str.split()
print(x)
x12 = x.str.split()
# print(x12)
# traindata_token = word_tokenize(x)
traindata_token = x.to_dict()

#%% ----------- Tokenize stop words ------------
s = stopwords['stopwords']
s = s.to_list()
print(s)

#%%
print(x[1])

youla = []

for w in x[1]:
    if w not in s:
        youla.append(w)
print()
print(youla)

#%% test
# if list of returned filtered list does not contain any words in stopwords list, return "we are good", else fix!! 

if "big" in s:
    print('yes')

else:
    print('no')


#%% -----------Remove stop words ------------
words = []

filtered_sentence = [w for w in traindata_token if not w in s] 

for w in traindata_token: 
    if w not in s: 
        filtered_sentence.append(w) 

print(traindata_token)
print(filtered_sentence) 





#%% --------------
stopwords.head()

trainlabels.head()

traindata.head()

testlabels.head()
#%%
traindata.head()
#%%
df = x
stop = stopwords

for entry in stop.stopwords:
    print(entry)

#%%

# z = lambda x: 
df = df.apply(
    lambda x: [item for item in x if item not in stop])

#%%

#%%

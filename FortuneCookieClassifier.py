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
# traindata_token = word_tokenize(x)
# traindata_token = x.to_dict()
traindata_token = x.to_frame()
print(type(traindata_token))
print(traindata_token)
#%% ----------- Tokenize stop words ------------
s = stopwords['stopwords']
s = s.to_list()
print(s)
#%% -----------Remove stop words ------------
the_vocab = traindata_token['traindata'].apply(lambda x: [item for item in x if item not in s])
the_vocab = the_vocab.to_frame()
print(the_vocab)
#%% ----------Sort each row in alphabetial order -----------
the_vocab.columns = ['vocab']

sorted_vocab = the_vocab['vocab'].apply(sorted)
print(sorted_vocab)

#%%

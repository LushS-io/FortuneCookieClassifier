# %%
import collections
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
import collections
from sklearn.linear_model import Perceptron
import sys

print('pandas version: {}'.format(pd.__version__))

# %% import datasets
# cwd = os.getcwd()
# traindata = pd.read_csv(filepath_or_buffer="./fortunecookiedata/traindata.txt",header=None,names=['traindata'])
# trainlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/trainlabels.txt',names=['trainlabels'])

# stopwords = pd.read_csv(filepath_or_buffer="./fortunecookiedata/stoplist.txt",header=None,names=['stopwords'])

# testdata = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testdata.txt',header=None,names=['testdata'])
# testlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testlabels.txt',header=None,names=['testlabels'])

# ---------option to use personal public s3 buckets to pull data -------------
traindata = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/traindata.txt", header=None, names=['traindata'])
trainlabels = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/trainlabels.txt", names=['trainlabels'])

stopwords = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/stoplist.txt", header=None, names=['stopwords'])
testdata = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/testdata.txt",header=None,names=['testdata'])
testlabels = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/testlabels.txt", header=None, names=['testlabels']

)

'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
# %% ----------- pre-processing -----------
# traindata.columns = ['traindata']
traindata.head()
traindata.shape
# %% --------------Tokenize traindata-----------
x = traindata['traindata'].str.split()
# traindata_token = word_tokenize(x)
# traindata_token = x.to_dict()
traindata_token = x.to_frame()
print(type(traindata_token))
print(traindata_token)
#%% ----------- Tokenize stop words ------------
s = stopwords['stopwords'].tolist()  # flatten to list
print(s)
#%% -----------Remove stop words ------------
the_vocab = traindata_token['traindata'].apply(lambda x: [item for item in x if item not in s])
the_vocab = the_vocab #-------the_vocab is trainining data without stopwords
print(the_vocab)
print(type(the_vocab))
#%% ----------Sort each row in alphabetial order -----------
sorted_vocab = the_vocab.apply(sorted)
# print(sorted_vocab)
print(type(sorted_vocab))
#%% ------------Feature extraction --------------
vocab = []
sorted_vocab.apply(lambda x: [vocab.append(word) for word in x if word not in vocab])
# print(vocab)

#%% -----------Sort vocab -----------------
vocab = sorted(vocab)
print(vocab)
# pd.DataFrame(vocab).shape

#%% -------------- check for duplicates ------------
dupe_list = vocab 
print([item for item, count in collections.Counter(dupe_list).items() if count > 1])

#%% ------------- Vectorize ------------- 
np.set_printoptions(threshold=sys.maxsize)
train_data_corpus = the_vocab.apply(func=lambda x: ' '.join(x))

vectorized = CountVectorizer()

vector = vectorized.fit_transform(train_data_corpus).todense()


# vectorized.fit(train_data_corpus)
# print(vectorized.vocabulary_)
# vector = vectorized.transform(train_data_corpus)

print(vector.shape)
print(type(vector))
# print(vector.toarray())
print(vector)

#%% --- check that shapes are correct ---
print(vector.shape)
print(testlabels.shape)
print(type(vectorized))
print(type(testlabels))

#%% --- see vectorized ---
print(vector)

# %% -------------- perform perceptron ---------------

# --- check shape ---
D = vector.shape[1] # D = 
print(D)

#%% --- create series of empty weights --- 
# weight vector should be the length of vocabulary

w = pd.Series(data=[0] * vector.shape[1])
print(w.size)
print(w)

#%%
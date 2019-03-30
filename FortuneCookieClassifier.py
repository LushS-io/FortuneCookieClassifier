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
s = s.to_list() #flatten into list 
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
new = the_vocab.apply(func=lambda x: ' '.join(x))

vec = CountVectorizer()
vec = vec.fit_transform(new).todense()
# print(vec.vocabulary_)

print(vec.shape)
print(type(vec))


#%%



# print(type(vocab))
corpus = [
    'All my cats in a row',
    'When my cat sits down, she looks like a Furby toy!',
    'The cat from outer space',
    'Sunshine loves to sit like this for some reason.'
]

print(type(corpus))

vocab_vectorized = CountVectorizer()
vocab_vectorized.fit_transform(the_vocab).todense()
#%%
# vocab_vectorized.fit(x)
print(vocab_vectorized.vocabulary_)
# vector = vocab_vectorized.transform(x)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# %% -------------- perform perceptron ---------------
Perceptron(vector)

#%%

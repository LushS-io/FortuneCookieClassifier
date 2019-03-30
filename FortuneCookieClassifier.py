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
the_vocab = the_vocab
print(the_vocab)
print(type(the_vocab))

#%% last play

new = the_vocab.apply(func = lambda x: ' '.join(x))

vec = CountVectorizer()
vec = vec.fit_transform(new).todense()
# print(vec.vocabulary_)

print(vec.shape)
print(type(vec))
#%% ----play ----
playcab = [
    'abc',
    'devbsdf',
    'lskdjfkjlkjsdf',
    'lksdflkj'
]

print(playcab)


u = the_vocab
u = u.loc[0]
u = ' '.join(u)

um = []
um.append(u)

print(um)

# vec = CountVectorizer()
# vec.fit_transform(u)
# u = the_vocab['traindata']
# u = the_vocab.loc[0]

## working with a series of  list 



# pd.concat(the_vocab['vocab'])
# u = u.to_string()

# u = ['ab','cd','ef']
# hu = ','

# u = ' '.join(u)

# u = str.join()
# print(type(u))
# print(type(u))

#%% play 2
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
print(s1)

#%% ----------Sort each row in alphabetial order -----------
the_vocab.columns = ['vocab']

sorted_vocab = the_vocab['vocab'].apply(sorted).to_frame()
print(sorted_vocab)

#%% ------------Feature extraction --------------
vocab = []
sorted_vocab['vocab'].apply(lambda x: [vocab.append(word) for word in x if word not in vocab])

#%% -----------Sort vocab -----------------
vocab = sorted(vocab)
print(vocab)
# pd.DataFrame(vocab).shape


#%% -------------- check for duplicates ------------
a = vocab 
print([item for item, count in collections.Counter(a).items() if count > 1])

#%% ------------- Vectorize ------------- 
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

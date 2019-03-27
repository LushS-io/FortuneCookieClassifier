# %%
import pandas as pd
import numpy as np
import os
import sklearn

# %% import datasets
cwd = os.getcwd()
traindata = pd.read_csv(filepath_or_buffer="./fortunecookiedata/traindata.txt",header=None)
trainlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/trainlabels.txt')

stopwords = pd.read_csv(filepath_or_buffer="./fortunecookiedata/stoplist.txt",header=None)

testdata = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testdata.txt',header=None)
testlabels = pd.read_csv(filepath_or_buffer='./fortunecookiedata/testlabels.txt',header=None)


'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
# %% pre-processing
traindata.columns = ['traindata']
traindata.head()
x = traindata['traindata'].str.split()
print(x)

stopwords.columns = ['stopwords']
stopwords.head()

trainlabels.columns = ['trainlabels']
trainlabels.head()

testdata.columns = ['traindata']
traindata.head()

testlabels.columns = ['trainlabels']
testlabels.head()
#%%
splitTrain = traindata['traindata'].str.split()
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

# %%
import pandas as pd
import numpy as np
import sklearn as sk
import os

# %% import datasets
cwd = os.getcwd()
traindata = pd.read_csv(filepath_or_buffer=cwd+"/FortuneCookieClassifier/fortunecookiedata/traindata.txt",header=None)
trainlabels = pd.read_csv(filepath_or_buffer=cwd+'/FortuneCookieClassifier/fortunecookiedata/trainlabels.txt',header=None)

stopwords = pd.read_csv(filepath_or_buffer=cwd+"/FortuneCookieClassifier/fortunecookiedata/stoplist.txt",header=None)
print(stopwords[0][5])

testdata = pd.read_csv(filepath_or_buffer=cwd+'/FortuneCookieClassifier/fortunecookiedata/testdata.txt',header=None)
testlabels = pd.read_csv(filepath_or_buffer=cwd + '/FortuneCookieClassifier/fortunecookiedata/testlabels.txt',header=None)


'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
# %% pre-processing
traindata.columns = ['trainData']
traindata.head()

stopwords.columns = ['stopwords']
stopwords.head()

trainlabels.columns = ['trainLabels']
trainlabels.head()

testdata.columns = ['trainData']
traindata.head()

testlabels.columns = ['trainLabels']
testlabels.head()
#%%
gah = traindata['trainData'].str.split()

#%%
gah = gah.apply(lambda x: [item for item in x if item not in stop])

gah.head()

#%%

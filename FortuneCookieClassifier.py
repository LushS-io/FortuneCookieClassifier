# %%
import pandas as pd
import numpy as np
import sklearn as sk
# %% import datasets
traindata = pd.read_csv('./fortunecookiedata/traindata.txt',header=None)
trainlabels = pd.read_csv('./fortunecookiedata/trainlabels.txt')

stopwords = pd.read_csv("./fortunecookiedata/stoplist.txt")

testdata = pd.read_csv('./fortunecookiedata/testdata.txt')
testlabels = pd.read_csv('./fortunecookiedata/trainlabels.txt')
# %% pre-processing
'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
traindata.columns = ['messages']
traindata.head()
# traindata[0].apply(lambda x: [item for item in x if item not in stopwords])
#%%


# %%
from sklearn.datasets import load_digits
import collections
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse 
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# import nltk
# from nltk.tokenize import word_tokenize
from sklearn.linear_model import Perceptron
import sys

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=1000)

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
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/testdata.txt", header=None, names=['testdata'])
testlabels = pd.read_csv(
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/testlabels.txt", header=None, names=['testlabels']

)

'''Create a dictionary of words in the training data. Utilize the stopwords list to remove insignificant words.'''
# %% ----------- exploratory data  analysis -----------
# traindata.columns = ['traindata']
traindata.head()
print(traindata.shape)
print(trainlabels.shape)
# %% --------------Tokenize traindata-----------
x = traindata['traindata'].str.split()
# traindata_token = word_tokenize(x)
# traindata_token = x.to_dict()
traindata_token = x.to_frame()
print(type(traindata_token))
print(traindata_token)
# %%
# ------------- Tokenize testdata --------
x_test = testdata['testdata'].str.split()
testdata_token = x_test.to_frame()
print(type(testdata_token))
print(testdata_token)

# %% ----------- Tokenize stop words ------------
s = stopwords['stopwords'].tolist()  # flatten to list
print(s)
# %% -----------Remove stop words ------------
the_vocab = traindata_token['traindata'].apply(
    lambda x: [item for item in x if item not in s])
the_vocab = the_vocab  # -------the_vocab is trainining data without stopwords
print(the_vocab)
print(type(the_vocab))

# ------------- for test data ------  repeat
# the_vocab_test = x_test.apply(lambda x: [item for item in x if item not in s])
# print(the_vocab_test)

# %% ----------Sort each row in alphabetial order -----------
sorted_vocab = the_vocab.apply(sorted)
# sorted_vocab_test = the_vocab_test.apply(sorted) # makes no sense 
print(sorted_vocab)
print(type(sorted_vocab))
# %% ------------Feature extraction --------------
vocab = []
sorted_vocab.apply(lambda x: [vocab.append(word)
                              for word in x if word not in vocab])
print(vocab)
# %% -----------Sort vocab ----------------- **** lol why am I sorting alphabetically twice?
vocab = sorted(vocab)
print(vocab)
# pd.DataFrame(vocab).shape # (693,1) ... 693 words

# %% -------------- check for duplicates ------------
dupe_list = vocab
print([item for item, count in collections.Counter(dupe_list).items() if count > 1]) # fix
if not dupe_list:
    print("No dupes!")

# %% ------------- Vectorize -------------
train_data_corpus = the_vocab.apply(func=lambda x: ' '.join(x))
print(train_data_corpus)

#%% --- Vectorize test data ---------
test_data_corpus = testdata_token['testdata'] # remove header 
test_data_corpus = test_data_corpus.apply(func=lambda x: ' '.join(x))
print(test_data_corpus)

# %% ---- use CountVectorizer to vectorize ----
vectorized = CountVectorizer()

# --- get training data in order ---
train_data_corpus_vectorized = vectorized.fit_transform(train_data_corpus).todense()
print(train_data_corpus_vectorized)
print(type(train_data_corpus_vectorized))
print(train_data_corpus_vectorized.shape)

# --- get test data in order ---
test_data_corpus_vectorized = vectorized.fit_transform(test_data_corpus).todense()
print(test_data_corpus_vectorized)
print(type(test_data_corpus_vectorized))
print(test_data_corpus_vectorized.shape)

# %% --- back to normal csr matrix --- 
test_data_sp_vectorized = sparse.csr_matrix(test_data_corpus_vectorized)
# print(test_data_sp_vectorized)

train_data_sp_vectorized = sparse.csr_matrix(train_data_corpus_vectorized)
# print(train_data_sp_vectorized)


# %%
# print(the_vocab_test)
# the_vocab_test = pd.Series(the_vocab_test['traindata'])
# test_data_corpus = the_vocab_test.apply(func=lambda x: ' '.join(x))
# print(test_data_corpus)
# ************* may still need to remove the stop words **************

# vectorized.fit(train_data_corpus)
# print(vectorized.vocabulary_)
# vector = vectorized.transform(train_data_corpus)

# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
# print(vector)

# %% --- check that shapes are correct ---
# print(vector.shape)
# print(trainlabels.shape)
# print(type(vectorized)) #vectorized is the train_data with removed stop and vectorized
# print(type(trainlabels)) # the training labels

# ----------
# print(testlabels.shape)

# %% --- see vectorized ---
# print(vector)
# vew = vector.tolist()
# print(vew)

# %% ---- Perceptron Functions -------

def my_predict(example, weight): # example = one row from training data  && weigtht = the weight vector we are training
    y_hat = 0 # give the initial 0 from first weight
    # x_i = example.tolist() # remove matrix layer
    # x_i = list(x_i[0]) # remove list of list layer to just  list
    # x_i = np.array(x_i) # Make np array again
    # w_i = np.array(weight) # make weight np array
    # print()

    # print(x_i)
    # print(w_i)

    y_hat = np.dot(example,weight) # compute dot product ... y_hat = x_i * w_i

    return y_hat # should be a scalar


def mistake_check(y_hat, label):
    label_floated = label.astype(float)
    if y_hat == label_floated:
        return True
    else:
        return False

def update(weight_old, learning_rate, train_label, train_features,y_hat):
    status = False # assume label is wrong
    status = mistake_check(y_hat,train_label)
    
    # # fix train_features ... now declared as x_i
    # x_i = train_features.tolist() # remove matrix layer
    # x_i = list(x_i[0]) # remove list of list layer to just list
    # x_i = np.array(x_i) # Make np array again
    # #

    if status:
        return weight_old # weight predicted the correct label
    else:
        weight_updated = weight_old + learning_rate * train_label * train_features 
        # weight_updated = weight_updated.T # maynot need.... 
    return weight_updated

def my_Perceptron(train, train_label, n_epoch, learning_rate=1):
    weight = np.zeros(train.shape[1])  # init weights
    weight = weight.T # this transformation may not matter
    n = 0 # counter for number of epoch interations gone through --- on watch list 
    for epoch in range(n_epoch):  # for each training iter
        i = 0 # counter for every row in traininig data
        n+=1 #update epooch
        for example in train:  # for each traiing example
            prediction = my_predict(example, weight)  # run predict
            weight = update(weight,learning_rate,train_label[i][0],example,prediction)  # if mistake update weight with return from update()
            i+=1 # update row predicted
    print("ran for {} epochs".format(n))
    return weight  # final weight


# %% ---- Train Weight Vector ----

# --- before reshape into np arrays --- easier to work with
X1 = np.array(train_data_sp_vectorized.todense())
y1 = np.array(trainlabels)

the_sauce = my_Perceptron(train=X1, train_label=y1, n_epoch=20)

print('Success, sauce made! \n\n {}'.format(the_sauce)) # :)

# %% -- Test weight on test data ---

# --- look at data ---
# print(testdata.shape)
# print(testlabels.shape)
# print(the_sauce.shape)

#%
def test_predict (messages,the_sauce):
    y_hat = 0  # give the initial 0 from first weight
    # x_i = message.tolist() # remove matrix layer
    # x_i = list(x_i[0]) # remove list of list layer to just  list
    # x_i = np.array(messages) # Make np array again
    # w_i = np.array(the_sauce) # make weight np array

    # print(x_i)
    # print(w_i)

    y_hat = np.dot(messages,the_sauce) # compute dot product ... y_hat = x_i * w_i
    # print(y_hat)

    return y_hat # should be a scalar

def test (test_data,test_labels,the_sauce):
    ding = False # assume mistake 
    i = 0 # test_label counter
    mistake_counter = 0 # counter number of mistakes
    correct_counter = 0 # count number of correct predictions
    for row in test_data: # loop test_data
        prediction = test_predict(row,the_sauce) # returns y_hat prediction
        ding = mistake_check(prediction,test_labels[i][0]) # capture if true or false
        i += 1 # keep moving
        if ding:
            correct_counter += 1
        else:
            mistake_counter += 1
            
    accuracy = mistake_counter / test_labels.shape[0] # get final accuracy
    print('mistakes = {}'.format(mistake_counter))
    print('correct = {}'.format(correct_counter))
    return accuracy
#%% --- RUN TEST ---

# use these instead of weird vectors
X1 = np.array(train_data_sp_vectorized.todense())
y1 = np.array(trainlabels)

how_good = test(test_data=X1,test_labels=y1,the_sauce=the_sauce)
print('Accuracy is: {}'.format(how_good))
# %% ----------- SKLEARN -----------
# ---  get data into correct form to use Perceptron from sklearn
X1 = train_data_sp_vectorized 
X1 = np.array(train_data_sp_vectorized.todense())
print(type(X1))
# print(X1)
y1 = np.array(trainlabels).ravel()
print(type(y1))
# print(y1)

print('shapes')
print(X1.shape)
print(y1.shape)
# Run Perceptron on data
mad = Perceptron(max_iter=20, tol=1e-3)
mad.fit(X1, y1)
mad.score(X1, y1)


#%%

# %%
from sklearn.datasets import load_digits
import collections
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# import nltk
# from nltk.tokenize import word_tokenize
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

# --------------
x_test = testdata['testdata'].str.split()
x_test = x.to_frame()
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
the_vocab_test = x_test.apply(lambda x: [item for item in x if item not in s])
print(the_vocab_test)

# %% ----------Sort each row in alphabetial order -----------
sorted_vocab = the_vocab.apply(sorted)
sorted_vocab_test = the_vocab_test.apply(sorted)
# print(sorted_vocab)
print(type(sorted_vocab))
# %% ------------Feature extraction --------------
vocab = []
sorted_vocab.apply(lambda x: [vocab.append(word)
                              for word in x if word not in vocab])
# print(vocab)
# %% -----------Sort vocab -----------------
vocab = sorted(vocab)
print(vocab)
# pd.DataFrame(vocab).shape

# %% -------------- check for duplicates ------------
dupe_list = vocab
print([item for item, count in collections.Counter(dupe_list).items() if count > 1])

# %% ------------- Vectorize -------------
np.set_printoptions(threshold=sys.maxsize)
train_data_corpus = the_vocab.apply(func=lambda x: ' '.join(x))

print(train_data_corpus)

#%% --- 

vectorized = CountVectorizer()

# --- get training data in order ---
train_data_corpus_vectorized = vectorized.fit_transform(train_data_corpus).todense()
# print(train_data_corpus_vectorized)
print(type(train_data_corpus_vectorized))
print(train_data_corpus_vectorized.shape)

# %%
# -----------
print(the_vocab_test)
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

# %% ---- Perceptron functions -------
# %% --- start another perceptron iteration from scratch ---


def flatten(l): return [item for sublist in l for item in sublist]

# def my_predict(example, weight):
#     y_hat = weight[0]
#     for i in range(len(example)-1):
#         y_hat = y_hat + weight[i + 1] * example[i]
#     return 1.0 if y_hat >= 0.0 else 0.0 # if y_hat is >=0 then 

def my_predict(example, weight): # example = one row from training data  && weigtht = the weight vector we are training
    y_hat = weight[0] # give the initial 0 from first weight
    x_i = example.tolist() # remove matrix layer
    x_i = list(x_i[0]) # remove list of list layer to just  list
    x_i = np.array(x_i) # Make np array again
    w_i = np.array(weight) # make weight np array

    print()

    # print(x_i)
    # print(w_i)

    y_hat += np.dot(x_i,w_i) # compute dot product ... y_hat = x_i * w_i

    return y_hat


def mistake_check(y_hat, label):
    label_floated = label.astype(float)
    if y_hat == label_floated:
        return True
    else:
        return False

def update(weight_old, learning_rate, train_label, train_features,y_hat):
    status = False#assume label is wrong
    status = mistake_check(y_hat,train_label)
    if status:
        return weight_old # weight needs to update
    else:
        weight_updated = weight_old + learning_rate * train_label * train_features 
        weight_updated = weight_updated.T
    return weight_updated

def my_Perceptron(train, train_label, n_epoch, learning_rate=1):
    weight = np.zeros(train.shape[1])  # init weights
    # print(weight.shape)
    weight = weight.T # this transformation may not matter
    # print(weight.shape)
    n = 0 # counter for number of epoch interations gone through --- on watch list 
    for epoch in range(n_epoch):  # for each training iter
        i = 0 # counter for every row in traininig data
        n+=1 #update epooch
        for example in train:  # for each traiing example
            prediction = my_predict(example, weight)  # run predict
            i+=1 # update row predicted
            error = train_label['trainlabels'].loc[i] - prediction# calculate error : error = expected Y - given Y(prediction)
            # print('prediction = {} || {}'.format(prediction,i))
            if error > 0:
                weight = update(weight,learning_rate,train_label['trainlabels'].loc[i],example,prediction)  # if mistake update weight with return from update()
            else:# run update
                print("hit something")
    return weight  # final weight


# %% ---- RUN ----
print(train_data_corpus_vectorized.shape)
print(trainlabels.shape)
my_Perceptron(train=train_data_corpus_vectorized, train_label=trainlabels, n_epoch=20)

# every example against the weight vector should give the proper train label...else mistake and update weights

# %%
#region - -----------*******   Sklearn  ------------------ ************ -----------------
# %% example to learn from  ///////////////////////// using sklearn

X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=None, n_iter=None, n_iter_no_change=5,
           n_jobs=None, penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
clf.score(X, y)  # doctest: +ELLIPSIS
# /////////////////////////

# %% get data into correct form to use Perceptron from sklearn
X1 = train_data_corpus_vectorized
# print(type(X1))
# print(vector)
y1 = np.array(trainlabels).ravel()
print(type(y1))

# %% Run Perceptron on data
mad = Perceptron(max_iter=20, tol=1e-3)
mad.fit(X1, y1)
mad.score(X1, y1)

# %%
print(testlabels.shape)
print(testdata.shape)
print(type(testdata))
print(type(testlabels))

# clf.score(X1,y1)

# endregion

# %% --- from  the web ---


def predict(row, weights):
    activation = weights[0]  # bias?
    # print('weights at 0 = {}'.format(activation)) # not sure what I'm checking here...
    for i in range(len(row)-1):  # for every weight
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>iteration_epoch=%d, learning_rate=%.3f, error=%.3f' %
              (epoch, l_rate, sum_error))
    return weights


# %% --- Run Perception --- from web
l_rate = 1  # learning rate
n_epoch = 20  # number of training iterations
# weights = train_weights(vew, l_rate, n_epoch)  # training step
# print("weight: {}".format(weights))  # print

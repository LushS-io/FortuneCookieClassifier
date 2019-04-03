# %%
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
    "https://s3-us-west-2.amazonaws.com/fortunecookie-dataset/fortuneCookie_data/testdata.txt",header=None,names=['testdata'])
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
#%% ----------- Tokenize stop words ------------
s = stopwords['stopwords'].tolist()  # flatten to list
print(s)
#%% -----------Remove stop words ------------
the_vocab = traindata_token['traindata'].apply(lambda x: [item for item in x if item not in s])
the_vocab = the_vocab #-------the_vocab is trainining data without stopwords
print(the_vocab)
print(type(the_vocab))

# -------------
the_vocab_test = x_test.apply(lambda x: [item for item in x if item not in s])
print(the_vocab_test)

#%% ----------Sort each row in alphabetial order -----------
sorted_vocab = the_vocab.apply(sorted)
sorted_vocab_test = the_vocab_test.apply(sorted)
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

print((the_vocab_test))

vectorized = CountVectorizer()

vector = vectorized.fit_transform(train_data_corpus).todense()

# -----------
# print(the_vocab_test)
the_vocab_test = pd.Series(the_vocab_test['traindata'])
test_data_corpus = the_vocab_test.apply(func=lambda x: ' '.join(x))


# ------------

# vectorized.fit(train_data_corpus)
# print(vectorized.vocabulary_)
# vector = vectorized.transform(train_data_corpus)

print(vector.shape)
print(type(vector))
# print(vector.toarray())
print(vector)

#%% --- check that shapes are correct ---
print(vector.shape)
print(trainlabels.shape)
print(type(vectorized))
print(type(trainlabels))

#----------
print(testlabels.shape)

#%% --- see vectorized ---
# print(vector)
vew = vector.tolist()
print(vew)

#%% ---- Perceptron functions -------
def predict(row, weights):
	activation = weights[0] #bias?
	# print('weights at 0 = {}'.format(activation)) # not sure what I'm checking here...
	for i in range(len(row)-1): #for every weight 
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
		print('>iteration_epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

#%% --- Run Perception ---
l_rate = 1 #learning rate
n_epoch = 20  # number of training iterations
weights = train_weights(vew, l_rate, n_epoch) #training step
print("weight: {}".format(weights)) #print 

#%% --- start another perceptron iteration from scratch ---
def my_predict(example, weight):
    y_hat = weight[0]
    for i in range(len(example)-1):
        y_hat = y_hat + weights[i + 1] * example[i]
    return 1.0 if y_hat >= 0.0 else 0.0

def update(weight, learning_rate, train_label, train):
    w = weight + learning_rate * train_label * train
    return w

def mistake_check(x):
    pass

def my_Perceptron(train, train_label, n_epoch, learning_rate = 1):
	weight = np.zeros(vector.shape[0]) # init weights
	for epoch in range(n_epoch): # for each training iter
	    for example in train:# for each traiing example
		    my_predict(example, weight)# run predict
	if mistake_check() > 0: # if mistake
		print("mistake found")# run update	
	return weight #final weight
#%%
my_Perceptron(train=vector,train_label=trainlabels, n_epoch=20)


#%% Angeleca Code


def step(example, w):
    '''
    '''
    y_hat = np.sign(np.dot(w, example))

    if y_hat.all() >= 0:
         return 1
    else:
        return 0


def perceptron(train, train_label, iter_t):
    '''
    #training example = train
    iter_t = maxmum number of training iterations
    output = w, final weight vector
    '''
    lr = 1  # learn rate
    #print(train.shape)
    w = [0.0 for i in range(train.shape[1])]
    #print(w)#initalize weight vector
    mistake = [0] * iter_t
    for x in range(iter_t):

        for e, example in enumerate(train):

            sf = step(example.T, w[e])

            #print(sf)
            if sf != int(train_label.iloc[e]):
                mistake[x] += 1
##
                xy = np.dot(int(train_label.iloc[e]), example)

                w[e] = w[e] + lr * xy
                #print(w[e])
#

    print(mistake)
    return w

#%% -- running

trainlabels.shape
perceptron(vector,trainlabels,20)

#%%
#region - Sklearn 
#%% example to learn from  ///////////////////////// using sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=None, n_iter=None, n_iter_no_change=5,
           n_jobs=None, penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
clf.score(X, y)  # doctest: +ELLIPSIS
 # /////////////////////////

#%% get data into correct form to use Perceptron from sklearn
X1 = vector 
# print(type(X1))
# print(vector)
y1 = np.array(trainlabels).ravel()
print(type(y1))

#%% Run Perceptron on data
mad  = Perceptron(max_iter=20, tol=1e-3)
mad.fit(X1,y1)
mad.score(X1,y1)

#%%
print(testlabels.shape)
print(testdata.shape)
print(type(testdata))
print(type(testlabels))

# clf.score(X1,y1)

#endregion

#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling Challenge LSTM
# 
# **Mark Wilber**
# 
# The challenge here is to build a classifier for 56 FDA food safety violation categories, which are very unbalanced (sizes spanning more than 3 orders of magnitude). There are two components/features:
# 
# * a boolean, `FDAISCRITICAL`, indicating whether the violation is 'critical' or not
# * a description of the violation, `VIOCOMMENT`, which can range from 0 to 844 'words'
#   * (It is shown below, that the two instances with no comments can be safely dropped.)
# 
# This notebook generates TF-IDF features after extracting unigrams and bigrams, and trains models using logistic regression, random forest, linear SVC and complement Naive Bayes to compare f1 scores and training times.
# 
# <font color='darkgreen'>**As thise notebook is lengthy, readers will find it much easier to navigate with [Jupyter Nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) installed, and Table of Contents (2) selected:**</font>

# ## Preliminaries
# 
# **Next two lines are useful in the event of external code changes.**

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Python imports
# 
# **Next two lines are for pretty output for all prints in a Pandas cell, not just the last.**

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# **`DataSci` contains generally helpful data science stuff, while `plotHelpers` includes plot functions specifically.**

# In[3]:


import sys
# sys.path.append('/home/wilber/work/Mlib')
sys.path.append('/home/mark/work/Mlib')
from utility import DataSci as util
from plotHelpers import plotHelpers as ph


# In[4]:


from time import time, asctime, gmtime
print(asctime(gmtime()))

t0 = time()

# from platform import node
import os
from os.path import exists
# import shutil
# from glob import glob
from random import random
from collections import Counter, OrderedDict
import gc		# garbage collection module
import pathlib
import pprint
# import pickle
import timeit

print("Python version: ", sys.version_info[:])
print("Un-versioned imports:\n")
prefixStr = ''
if 'collections' in sys.modules:
    print(prefixStr + 'collections', end="")
    prefixStr = ', '
if 'gc' in sys.modules:
    print(prefixStr + 'gc', end="")
    prefixStr = ', '
if 'glob' in sys.modules:
    print(prefixStr + 'glob', end="")
    prefixStr = ', '
if 'pickle' in sys.modules:
    print(prefixStr + 'pickle', end="")
    prefixStr = ', '
if 'platform' in sys.modules:
    print(prefixStr + 'platform', end="")
    prefixStr = ', '
if 'plotHelpers' in sys.modules:
    print(prefixStr + 'plotHelpers', end="")
    prefixStr = ', '
if 'pprint' in sys.modules:
    print(prefixStr + 'pprint', end="")
    prefixStr = ', '
if 'os' in sys.modules:
    print(prefixStr + 'os', end="")
    prefixStr = ', '
if 'os.path' in sys.modules:
    print(prefixStr + 'os.path', end="")
    prefixStr = ', '
if 'random' in sys.modules:
    print(prefixStr + 'random', end="")
    prefixStr = ', '
if 'shutil' in sys.modules:
    print(prefixStr + 'shutil', end="")
    prefixStr = ', '
if 'sys' in sys.modules:
    print(prefixStr + 'sys', end="")
    prefixStr = ', '
if 'timeit' in sys.modules:
    print(prefixStr + 'timeit', end="")
    prefixStr = ', '
if 'utility' in sys.modules:
    print(prefixStr + 'utility', end="")
    # prefixStr = ', '

duVersion = None
from dateutil import __version__ as duVersion
from dateutil.parser import parse
import numpy as np
import pandas as pd
import pyreadr

scVersion = None
from scipy import __version__ as scVersion
import scipy.sparse as sp

skVersion = None
from sklearn import __version__ as skVersion
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import ComplementNB

tfVersion = None
from tensorflow import __version__ as tfVersion
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

jlVersion = None
from joblib import __version__ as jlVersion
from joblib import dump, load

import seaborn as sns
import colorcet as cc

mpVersion = None
from matplotlib import __version__ as mpVersion
import matplotlib.pyplot as plt

print("\n")
if 'colorcet' in sys.modules:
    print(f"colorcet: {cc.__version__}", end="\t")
if 'dateutil' in sys.modules:
    print(f"dateutil: {duVersion}", end="\t")
if 'joblib' in sys.modules:
    print(f"joblib: {jlVersion}", end="\t")
if 'matplotlib' in sys.modules:
    print(f"matplotlib: {mpVersion}", end="\t")
if 'numpy' in sys.modules:
    print(f"numpy: {np.__version__}", end="\t")
if 'pandas' in sys.modules:
    print(f"pandas: {pd.__version__}", end="\t")
if 'pyreader' in sys.modules:
    print(f"pyreader: {pyreader.__version__}", end="\t")
if 'scipy' in sys.modules:
    print(f"scipy: {scVersion}", end="\t")
if 'seaborn' in sys.modules:
    print(f"seaborn: {sns.__version__}", end="\t")
if 'sklearn' in sys.modules:
    print(f"sklearn: {skVersion}", end="\t")
if 'tensorflow' in sys.modules:
    print(f"tensorflow: {tfVersion}", end="\t")
# if '' in sys.modules:
#     print(f": {.__version__}", end="\t")
Δt = time() - t0
print(f"\n\nΔt: {Δt: 4.1f}s.")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Helper functions
# 
# <a id="helper-tokenize"></a>
# #### `tokenize()`

# In[5]:


def tokenize(corpus, vocabSz):
    """
    Generates the vocabulary and the list of list of integers for the input corpus

    Help from: https://www.tensorflow.org/tutorials/text/nmt_with_attention

    INPUTS:
        corpus: list, type(str), containing (short) document strings
        vocabSz: (int) Maximum number of words to consider in the vocabulary

    RETURNS: List of list of indices for each title in the corpus + Keras sentence tokenizer object

    Usage:
        listOfListsOfIndices, sentenceTokenizer = tokenize(mySentences, maxVocabCt)
    """

    # Define the sentence tokenizer
    sentenceTokenizer = Tokenizer(num_words=vocabSz,
                                  filters='!#%()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower=True,
                                  split=' ', char_level=False, oov_token="<unkwn>")

    # Keep the double quote, dash, and single quote + & (different from word2vec training: didn't keep `&`)
    # oov_token: added to word_index & used to replace out-of-vocab words during text_to_sequence calls
    # num_words = maximum number of words to keep, dropping least frequent

    # Fit the tokenizer on the input corpus
    sentenceTokenizer.fit_on_texts(corpus)

    # Transform each text in corpus to a sequence of integers
    listOfIndexLists = sentenceTokenizer.texts_to_sequences(corpus)

    return listOfIndexLists, sentenceTokenizer


# #### `df2TFdata()`
# 
# This is modified from [a TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/feature_columns), replacing columnar feature with tokenized text for the inputs.

# In[6]:


# def df2TFdata(dataframe, textCol, targetCol, shuffle=True, batchSz=64):
#     """
#     dataframe		pd.DataFrame, containing a column with text, and a target column with labels
#     shuffle			bool, indicating whether to shuffle the data, default: True
#     batchSz			int, indicating batch size, default: 64
#     """

#     dataframe = dataframe.copy()
#     labels = dataframe.pop(targetCol)
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)

#     return ds


# ## Handle the data
# 
# ### Read data into a DataFrame
# 
# * Have a very quick look at DataFrame characteristics

# In[7]:


fname = "SelectedInspectionReportData.rds"

t0 = time()
result = pyreadr.read_r(fname)
df = result[None]
df.fda_q_fixed = df.fda_q_fixed.astype('int')
df.FDAISCRITICAL = df.FDAISCRITICAL.astype('int')
Δt = time() - t0
print(f"\n\nΔt: {Δt: 4.1f}s.")

df.shape


# In[8]:


df.head(6).T
df.tail(6).T


# #### Basic summary

# In[9]:


df.info()


# In[10]:


df.describe()


# #### Remove columns from DataFrame which we won't need

# In[11]:


df = df[['fda_q_fixed', 'VIOCOMMENT', 'FDAISCRITICAL']]
df.info()


# ### Exploratory analysis
# 
# #### Classes and relative balance
# 
# * The stuff using patches is for placing counts above each rectangle in the bar plot

# In[12]:


FDAcodes = list(set(df['fda_q_fixed'].values))
print(FDAcodes)
classCts = pd.DataFrame(df['fda_q_fixed'].value_counts())
with pd.option_context("display.max_columns", 60):
    display(classCts.T)


# In[13]:


ph.plotValueCounts(df, 'fda_q_fixed', titleText='FDA code frequencies', saveAs='svg', ylim=[0.0, 187500.0])


# ***The class sizes span nearly 4 orders of magnitude!***

# #### Word frequencies

# In[14]:


t0 = time()
df['commentsWords'] = df['VIOCOMMENT'].apply(lambda s: s.split())
t1 = time()
Δt = t1 - t0
print(f"Δt: {Δt % 60.0:4.1f}s.")


# In[15]:


comments = list(df['commentsWords'])
print(comments[0])
print(comments[-1])


# ##### Distribution of comment lengths
# 
# * Add length of each comment to DataFrame as `wordFreq` column

# In[16]:


wordLens = [len(wordList) for wordList in comments]
df['wordFreq'] = wordLens
wordFreqMode = df['wordFreq'].mode().values[0]

wordCtSorted = sorted(wordLens)
print("smallest word counts:\n", wordCtSorted[:100])
print("largest word counts:\n", wordCtSorted[-101:-1])


# **Detailed histogram**

# In[17]:


fig, ax = plt.subplots(1, 1, figsize=(18, 3.5))

ph.detailedHistogram(wordLens, ylabel='frequency', volubility=2,
                     titleText=f"Word counts (max: {wordCtSorted[-1]}, mode: {wordFreqMode})",
                     figName="WordCountsHist", ax=ax, ylim = [0.5, 100000.0], ylog=True, saveAs='svg')


# **Make space**

# In[18]:


del wordLens
del wordCtSorted
del df['commentsWords']


# ##### What FDA codes correspond to those comments having `wordFreq== 0`?

# In[19]:


df[df['wordFreq']==0]
print("\n", df.shape)


# **Can safely remove a couple of records from the 2nd-most populated category**
# 
# * Originally there were 1307986 records in `df`, out of which 122314 were in Class 49

# In[20]:


df = df[df['wordFreq']!=0]
df.shape


# ##### `wordFreq` percentiles
# 
# * These show that would get 99% coverage of the comments without truncation if were to use, say, 140-element LSTMs

# In[21]:


df.describe(percentiles=[0.01, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99])


# #### Most-common words

# In[22]:


allWords = [word for wordList in comments for word in wordList]		# Flatten list of lists of words
print(len(comments), len(allWords))

print(comments[:5], "\n", allWords[:25])


# In[23]:


t0 = time()
wordCtr = Counter(allWords)
t1 = time()
Δt = t1 - t0
print(f"Δt: {Δt % 60.0:4.1f}s.")


# ##### Most common words, after removing stop words
# 
# *Result looks very plausible*

# In[24]:


stopWords = text.ENGLISH_STOP_WORDS.union(['-'])

wcStops = [k for k in wordCtr if k.lower() in stopWords]
for k in wcStops:
    del wordCtr[k]
wordCtr.most_common(40)


# #### Clean up

# In[25]:


del allWords
del wordCtr


# #### `fda_q_fixed` vs. `FDAISCRITICAL`
# 
# What is the relationship between the critical violation boolean and the FDA code?

# In[26]:


dfCrit = df.groupby(['fda_q_fixed', 'FDAISCRITICAL']).count()
del dfCrit['VIOCOMMENT']
del dfCrit['wordFreq']
dfCrit.head(20)

dfCrit.reset_index(inplace=True)
dfCrit.head(20)


# In[27]:


plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labeltop'] = True

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
dfCrit.plot.scatter('fda_q_fixed', 'FDAISCRITICAL', s=4, c='black', ax=ax)
for xv in np.linspace(0.5, 56.5, 57):
    _ = plt.axvline(x=xv, c="#FFB0FF", linewidth=1)
plt.suptitle('Critical violations vs FDA code')
ax.set_xlim([0.5, 56.5])
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('CriticalViolationVsFDAcode.svg')


# **The critical violations plot shows that `FDAISCRITICAL` should be predictive (and certainly should be included in the model):**
# 
# * **<font color="darkgreen">classes 30, 32, 34 &amp; 46 *never* have critical violations</font>**
# * **<font color="darkgreen">classes 7, 26, 27 &amp; 29 *only* have critical violations</font>**

# ## Model Parameters
# 
# * testFrac				fraction of data set withheld
# * maxVocabCt			vocabulary size returned by Tokenizer, dropping least frequent
# * LSTMlayerUnits		# units within each activation unit in LSTMs]
# * embeddingDim			size of dimension for generated embeddings
# * auxFeaturesCt			# of features in auxiliary data
# * classCt				# classes (softmax output dim)
# * dropoutFrac			dropout fraction
# * LSTMdropoutFrac		dropout fraction within LSTMs
# * batchSz				size of batches
# * epochCt				number of epochs to run

# In[43]:


testFrac = 0.4
maxVocabCt = 80000
maxCommentWords = 140
LSTMlayerUnits = 64
embeddingDim = 64
auxFeaturesCt = 1
classCt = 56
dropoutFrac = 0.15
LSTMdropoutFrac = 0.5
batchSz = 64
epochCt = 10


# ## Pre-process data
# 
# * Here we use TF-IDF features to represent the comment text.
# 
# ### Split DataFrame by classes
# 
# * create a `numpy.random.RandomState` instance to keep track of the random number initializer, in order to ensure consistent results throughout
# 
# `splitDataFrameByClasses()` will create two new DataFrames, dfTr, dfTe, according to the desired splits.
# 
# <font color='darkgreen'><b>Note that if you just want to do stratified sampling on a numpy array of</b> `X` <b>values,</b> `splitDataFramByClasses()` <b>is not needed.</b> `train_test_split()` <b>accepts the keyword</b> `stratify=myTargetVariable`.</b></font>
# 
# * Splitting is done on a per-class basis, so that random selection will not, by chance, yield huge imbalances in train-test splits of tiny classes

# In[29]:


randomState=0
myRandomState = np.random.RandomState(randomState)


# In[30]:


classColumn = 'fda_q_fixed'
dfTr, dfTe = util.splitDataFrameByClasses(df, classColumn,
                                          testFrac=testFrac,
                                          myRandomState=myRandomState)
dfTr.shape, dfTe.shape
dfTr.head()
dfTe.head()


# **As intended, `splitBalancedDataFrameClasses()` created new test and train DataFrames, each with ~ 1307984/2 = 653992 rows.**
# 
# *The test DataFrame is not an exactly split of the original, since the splitting is done by class and unioned. For a 50% split, sci-kit learn's* `train_test_split()` gives the extra instance in each odd-sized class to the test set.*

# ### Create list of lists of word indices, and TensorFlow sentence tokenizer object
# 
# Use comment strings from `dfTrain` to create vocabulary indices.
# 
# See [helper function `tokenize()`](#helper-tokenize)

# In[31]:


ListOfCommentsTr = list(dfTr.VIOCOMMENT)

listOfListsOfWordIndicesTr, sentenceTokenizer = tokenize(ListOfCommentsTr, maxVocabCt)


# ### Pre-pad short comment lists, truncate the ends of long comments

# In[32]:


# padValue = max(max(listOfListsOfWordIndices)) + 1
padValue = 0
XcommentsTr = pad_sequences(listOfListsOfWordIndicesTr,
                            maxlen=maxCommentWords,
                            dtype='int32', padding='pre',
                            truncating='post', value=padValue)


# In[33]:


ListOfCommentsTr[0]
listOfListsOfWordIndicesTr[0]
XcommentsTr[0]


# ### Auxiliary (side) data need to be shaped

# In[34]:


XauxTr = dfTr.FDAISCRITICAL.values.reshape(dfTr.shape[0], 1)
XauxTr.shape


# ### Train target data

# In[35]:


FDAcodesTr = dfTr.fda_q_fixed - 1
print(set(FDAcodesTr))


# ### Tensor of word indices for test

# In[36]:


ListOfCommentsTe = list(dfTe.VIOCOMMENT)
listOfListsOfWordIndicesTe = sentenceTokenizer.texts_to_sequences(ListOfCommentsTe)
XcommentsTe = pad_sequences(listOfListsOfWordIndicesTe,
                            maxlen=maxCommentWords,
                            dtype='int32', padding='pre',
                            truncating='post', value=padValue)


# ### Auxiliary test data

# In[37]:


XauxTe = dfTe.FDAISCRITICAL.values.reshape(dfTe.shape[0], 1)
XauxTe.shape


# ### Test target data

# In[38]:


FDAcodesTe = dfTe.fda_q_fixed - 1
print(set(FDAcodesTe))


# ## Model time
# 
# ### Define the model
# 
# This follows, to some degree, [Keras Multi-Input and multi-output models](https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models)
# 
# * In this case, we only have a single output
# * Here, Bidirectional LSTMs are used

# In[40]:


def buildModel(sequence_length, vocabSz, auxFeatureCount, LSTMinternalLayerSz,
               embedLayerDim, densLayerDim=64, softMaxCt=16, dropoutFrac=0.15,
               LSTMdropoutFrac=0.40):

    """
    INPUTS:
    sequence_length				int, number of LSTM units
    vocabSz					int, size of vocabulary
    auxFeatureCount		int, count of auxiliary (side) features
    LSTMinternalLayerSz					int, size of layers within LSTM units
    embedLayerDim						int, dimension of embedding layer
    densLayerDim					int, dimension of dense layers, default: 64
    softMaxCt				int, dimension of softmax output, default: 16
    dropoutFrac				int, dropout rate, default: 0.50 [currently not used]
    """

    # Headline input: meant to receive sequences of *sequence_length* integers, between 1 and *vocabSz*.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(sequence_length,), dtype='int32', name='main_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 64-dimensional vectors.
    x = Embedding(output_dim=embedLayerDim, input_dim=vocabSz,
                  input_length=sequence_length, trainable=True)(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out_1 = Bidirectional(LSTM(LSTMinternalLayerSz,
                                    dropout=dropoutFrac,
                                    recurrent_dropout=LSTMdropoutFrac,
                                    return_sequences=True))(x)
    lstm_out = LSTM(LSTMinternalLayerSz,
                    dropout=dropoutFrac,
                    recurrent_dropout=LSTMdropoutFrac)(lstm_out_1)

    auxiliary_input = Input(shape=(auxFeatureCount,), name='numerical_input')
    x = concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(densLayerDim, activation='relu')(x)
    x = Dense(densLayerDim, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(56, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)
    
    return model


# ### Instantiate the model

# In[41]:


np.random.seed(0)  # Set a random seed for reproducibility
modelLSTM = buildModel(maxCommentWords, maxVocabCt, auxFeaturesCt,
                       LSTMlayerUnits, embeddingDim, softMaxCt=classCt)
modelLSTM.summary()


# ### Define callbacks for model checkpoints and TensorBoard

# In[47]:


checkpointDir = './checkpoints'

tensorBoardLogDir = './tensorBoardLogs'
os.makedirs(tensorBoardLogDir, exist_ok=True)

logDir = (f"vocabCt{maxVocabCt:06d}maxCommentLen{maxCommentWords:03d}auxFeaturesCt{auxFeaturesCt:02d}"
          + f"classCt{classCt:02d}embedDim{embeddingDim:03d}LSTMlayerSz{LSTMlayerUnits:03d}"
          + f"batchSz{batchSz:03d}dropoutFrac{dropoutFrac:4.2f}LSTMdropoutFrac{dropoutFrac:4.2f}")

logsDir = os.path.join(tensorBoardLogDir, logDir)

checkpointPrefix = os.path.join(checkpointDir, logDir, "ckpt{epoch:03d}")
checkpoint_callback=ModelCheckpoint(filepath=checkpointPrefix,
                                    save_weights_only=True)


os.makedirs(logsDir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logsDir, histogram_freq=1)


# ### Compile the model

# In[42]:


modelLSTM.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy', 'categorical_crossentropy'])


# ### Run the model

# In[ ]:


history = modelLSTM.fit(x=[XcommentsTr, XauxTr],
                    y= FDAcodesTr,
                    epochs=epochCt, batch_size=batchSz,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[checkpoint_callback, tensorboard_callback], verbose=1)


# In[ ]:


# history = modelLSTM.fit({'main_input': paddedSequences, 'numerical_input': bools},
#                         {'main_output': FDAcodes}, epochs=5, batch_size=batchSz)


# In[ ]:


modelLSTM.save(current_iteration + '/save')


# ### Do inference on test

# In[ ]:


softmaxOut = modelLSTM.predict(x=[XcommentsTe, XauxTe])


# In[ ]:


yPred = np.argmax(softmaxOut, axis=1) + 1


# In[ ]:


dfTe.head(3)
dfTe.shape


# #### Overall accuracy, precision, recall

# In[ ]:


yTe = dfTe.fda_q_fixed
confusionMat = confusion_matrix(yTe, yPred)
print(confusionMat)


# In[ ]:


get_ipython().run_line_magic('reset', 'out')


# In[ ]:


np.where(np.sum(confusionMat, axis=0) == 0)


# In[ ]:


accuracy = np.trace(confusionMat)/np.sum(confusionMat)
recall = np.diag(confusionMat)/np.sum(confusionMat, axis=1)
precision = np.diag(confusionMat)/np.sum(confusionMat, axis=0)
print(f"accuracy: {accuracy:0.3f}, "
      f"<precision>: {np.mean(precision):0.3f}, "
      f"<recall>: {np.mean(recall):0.3f}")


# ##### Recall, precision by class
# 
# Note:
# 
# * `macro avg`: $\frac{1}{K}\sum_k m_k$, where $K$ is count of classes and $m_k$ is a given metric for class $k$
# * `weighted avg`: $\frac{1}{N}\sum_k n_k \cdot m_k$, where $N$ is count of data instance, $n_k$ is the count of points in class $k$ and $m_k$ is a given metric for class $k$.

# In[ ]:


print(metrics.classification_report(yTe, yPred, target_names=[str(c)for c in FDAcodes]))


# In[ ]:


classCts = dfTe['fda_q_fixed'].value_counts()

recall = np.diag(confusionMat)/np.sum(confusionMat, axis = 1)
precision = np.diag(confusionMat)/np.sum(confusionMat, axis = 0)
f1 = 2.0*precision*recall/(precision + recall)
print("class\tprecision\trecall\tf1\tsize")

for FDAcode, classCt in classCts.iteritems():
    print(f"{FDAcode}\t{precision[FDAcode - 1]:0.3f}\t\t{recall[FDAcode - 1]:0.3f}\t{f1[FDAcode - 1]:0.3f}\t\t{classCt:d}")


# ##### Plot confusion matrix
# 
# * As this is a straight confusion matrix, diagonal elements mostly reflect class size in test set
# * *This is hard to interpret by visual inspection alone*

# In[ ]:


labelFontSz = 16
tickFontSz = 13
titleFontSz = 20


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(25, 25))
ph.plotConfusionMatrix(confusionMat, saveAs='pdf', xlabels=FDAcodes,
                       ylabels=FDAcodes, titleText = 'Logistic Regression',
                       ax = ax,  xlabelFontSz=labelFontSz,
                       ylabelFontSz=labelFontSz, xtickFontSz=tickFontSz,
                       ytickFontSz=tickFontSz, titleFontSz=titleFontSz)


# ##### Plot recall confusion matrix (normalized by row)
# 
# * diagonal elements now represent the *recall* for each class

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(25, 25))
ph.plotConfusionMatrix(confusionMat, saveAs='pdf', xlabels=FDAcodes, type='recall',
                       ylabels=FDAcodes, titleText = 'Logistic Regression',
                       ax = ax,  xlabelFontSz=labelFontSz,
                       ylabelFontSz=labelFontSz, xtickFontSz=tickFontSz,
                       ytickFontSz=tickFontSz, titleFontSz=titleFontSz)


# ##### Plot precision confusion matrix (normalized by column)
# 
# * diagonal elements now represent the *precision* for each class

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(25, 25))
ph.plotConfusionMatrix(confusionMat, saveAs='pdf', xlabels=FDAcodes, type='precision',
                       ylabels=FDAcodes, titleText = 'Logistic Regression',
                       ax = ax,  xlabelFontSz=labelFontSz,
                       ylabelFontSz=labelFontSz, xtickFontSz=tickFontSz,
                       ytickFontSz=tickFontSz, titleFontSz=titleFontSz)


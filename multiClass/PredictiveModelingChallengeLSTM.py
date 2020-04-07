#!/usr/bin/python3
# coding: utf-8

# # Predictive Modeling Challenge LSTM

# **Mark Wilber**

# The challenge here is to build a classifier for 56 FDA food safety violation
# categories, which are very unbalanced (sizes spanning more than 3 orders of
# magnitude). There are two components/features:

# * a boolean, `FDAISCRITICAL`, indicating whether the violation is 'critical'
#   or not a description of the violation, `VIOCOMMENT`, which can range from 0
#   to 844 'words'
# * (It is shown below, that the two instances with no comments can be safely
#   dropped.)

# This notebook generates TF-IDF features after extracting unigrams and
# bigrams, and trains models using logistic regression, random forest, linear
# SVC and complement Naive Bayes to compare f1 scores and training times.

# ## Preliminaries

import sys

from time import time, asctime, gmtime

# t0 = time()

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

# duVersion = None
from dateutil import __version__ as duVersion
from dateutil.parser import parse
import numpy as np
import pandas as pd
import pyreadr
import pydot_ng
import graphviz

# scVersion = None
from scipy import __version__ as scVersion
import scipy.sparse as sp

# skVersion = None
from sklearn import __version__ as skVersion
# from sklearn.feature_extraction import text
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import chi2
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.svm import LinearSVC, SVC
# from sklearn.naive_bayes import ComplementNB
from sklearn.utils import class_weight

# tfVersion = None
from tensorflow import __version__ as tfVersion
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Embedding, Bidirectional
from tensorflow.keras.layers import LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model

from joblib import __version__ as jlVersion
# from joblib import dump, load

import seaborn as sns
import colorcet as cc

from matplotlib import __version__ as mpVersion
import matplotlib.pyplot as plt

print("Python version: ", sys.version_info[:])
print("Un-versioned imports:\n")
prefixStr = ''
if 'collections' in sys.modules:
    print(prefixStr + 'collections', end="")
    prefixStr = ',  '
if 'gc' in sys.modules:
    print(prefixStr + 'gc', end="")
    prefixStr = ',  '
if 'glob' in sys.modules:
    print(prefixStr + 'glob', end="")
    prefixStr = ',  '
if 'pickle' in sys.modules:
    print(prefixStr + 'pickle', end="")
    prefixStr = ',  '
if 'platform' in sys.modules:
    print(prefixStr + 'platform', end="")
    prefixStr = ',  '
if 'pprint' in sys.modules:
    print(prefixStr + 'pprint', end="")
    prefixStr = ',  '
if 'os' in sys.modules:
    print(prefixStr + 'os', end="")
    prefixStr = ',  '
if 'os.path' in sys.modules:
    print(prefixStr + 'os.path', end="")
    prefixStr = ',  '
if 'random' in sys.modules:
    print(prefixStr + 'random', end="")
    prefixStr = ',  '
if 'shutil' in sys.modules:
    print(prefixStr + 'shutil', end="")
    prefixStr = ',  '
if 'sys' in sys.modules:
    print(prefixStr + 'sys', end="")
    prefixStr = ',  '
if 'timeit' in sys.modules:
    print(prefixStr + 'timeit', end="")
    prefixStr = ',  '

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

print("\n\n\t<<< !!! ------------------ !!! >>>\n\n"
      + "If next line breaks this script, invoke using:\n"
      + f"PYTHONPATH=$PYTHONPATH:/home/mark/work/Mlib {sys.argv[0]}\n\n"
      + "\t<<< !!! ------------------ !!! >>>\n")
# sys.path.append('/home/wilber/work/Mlib')
# sys.path.append('/home/mark/work/Mlib')
from utility import DataSci as util
from plotHelpers import plotHelpers as ph

prefixStr = ''
if 'plotHelpers' in sys.modules:
    print(prefixStr + 'plotHelpers', end="")
    prefixStr = ',  '
if 'utility' in sys.modules:
    print(prefixStr + 'utility', end="")

print(asctime(gmtime()))

# Δt = time() - t0
# print(f"\n\nΔt: {Δt: 4.1f}s.")

# ### Helper functions

# #### `tokenize()`

def tokenize(corpus, vocabSz):
    """
    Generates the vocabulary and the list of list of integers for the input
    corpus

    Help from: https://www.tensorflow.org/tutorials/text/nmt_with_attention

    INPUTS:
        corpus: list, type(str), containing (short) document strings
        vocabSz: (int) Maximum number of words to consider in the vocabulary

    RETURNS: List of list of indices for each title in the corpus + Keras
    sentence tokenizer object

    Usage:
        listOfListsOfIndices, sentenceTokenizer = \
            tokenize(mySentences, maxVocabCt)
    """

    # Define the sentence tokenizer
    sentenceTokenizer = Tokenizer(num_words=vocabSz,
                                  filters='!#%()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower=True,
                                  split=' ', char_level=False,
                                  oov_token="<unkwn>")

    # Keep the double quote, dash, and single quote + & (different from
    # word2vec training, which did not keep `&`)

    # oov_token: added to word_index & used to replace out-of-vocab words
    #            during text_to_sequence calls
    # num_words: maximum number of words to keep, dropping least frequent

    # Fit the tokenizer on the input corpus
    sentenceTokenizer.fit_on_texts(corpus)

    # Transform each text in corpus to a sequence of integers
    listOfIndexLists = sentenceTokenizer.texts_to_sequences(corpus)

    return listOfIndexLists, sentenceTokenizer


def preprocessDatums(df, classColumn, testFrac):

    """

    Splits data into train / test sets

    Use helper function `tokenizef()`, which invokes Keras Tokenizer, to
    tokenize sentences
    * This will return a list of lists, with each of the latter containing an
      index for each word in a comment
    * Finally the input text, `XcommentsTr`, is created by padding / truncating
      each comment index list to a length of 140.


    * create a `numpy.random.RandomState` instance to keep track of the random
      number initializer, in order to ensure consistent results throughout

    * `splitDataFrameByClasses()` will create two new DataFrames, dfTr, dfTe,
      according to the desired splits.
      * Splitting is done on a per-class basis, so that random selection will
        not, by chance, yield huge imbalances in train-test splits of tiny
        classes.

    ** Note that if you just want to do stratified sampling on a numpy array of
       `X` values, `splitDataFramByClasses()` is not needed.
       `train_test_split()` accepts the keyword `stratify=myTargetVariable`.
    **
    """

    randomState = 0
    myRandomState = np.random.RandomState(randomState)

    dfTr, dfTe = util.splitDataFrameByClasses(df, classColumn,
                                              testFrac=testFrac,
                                              myRandomState=myRandomState)
    print(f"dfTr.shape: {dfTr.shape}, dfTe.shape: {dfTe.shape}")
    print(dfTr.head().to_markdown())
    print(dfTe.head().to_markdown())

    # ### Create list of lists of word indices, and TensorFlow sentence
    # ### tokenizer object

    # Use comment strings from `dfTrain` to create vocabulary indices.
    # See [helper function `tokenize()`](#helper-tokenize)

    ListOfCommentsTr = list(dfTr.VIOCOMMENT)

    listOfListsOfWordIndicesTr, sentenceTokenizer = \
        tokenize(ListOfCommentsTr, maxVocabCt)

    # ### Pre-pad short comment lists, truncate the ends of long comments

    # padValue = max(max(listOfListsOfWordIndices)) + 1
    padValue = 0
    XcommentsTr = pad_sequences(listOfListsOfWordIndicesTr,
                                maxlen=maxCommentWords,
                                dtype='int32', padding='pre',
                                truncating='post', value=padValue)

    print(f"ListOfCommentsTr[0]: {ListOfCommentsTr[0]}")
    print(f"listOfListsOfWordIndicesTr[0]: {listOfListsOfWordIndicesTr[0]}")
    print(f"XcommentsTr[0]: {XcommentsTr[0]}")

    # ### Auxiliary (side) data need to be shaped

    XauxTr = dfTr.FDAISCRITICAL.values.reshape(dfTr.shape[0], 1)
    XauxTr.shape

    # ### Train target data

    FDAcodesTr = dfTr.fda_q_fixed - 1
    print(f"set(FDAcodesTr): {set(FDAcodesTr)}")

    # ### Tensor of word indices for test

    ListOfCommentsTe = list(dfTe.VIOCOMMENT)
    listOfListsOfWordIndicesTe = \
        sentenceTokenizer.texts_to_sequences(ListOfCommentsTe)
    XcommentsTe = pad_sequences(listOfListsOfWordIndicesTe,
                                maxlen=maxCommentWords,
                                dtype='int32', padding='pre',
                                truncating='post', value=padValue)

    # ### Auxiliary test data

    XauxTe = dfTe.FDAISCRITICAL.values.reshape(dfTe.shape[0], 1)
    XauxTe.shape

    FDAcodesTe = dfTe.fda_q_fixed - 1
    print(f"set(FDAcodesTe): {set(FDAcodesTe)}")

    return (dfTr, XcommentsTr, XauxTr, FDAcodesTr,
            dfTe, XcommentsTe, XauxTe, FDAcodesTe, myRandomState)

# Model time

# ### Define the model

# This follows, to some degree, [Keras Multi-Input and multi-output models]
# (https://keras.io/getting-started/...
#  .../functional-api-guide/#multi-input-and-multi-output-models)

# * In this case, we only have a single output
# * Here, Bidirectional LSTMs are used, but no secondary LSTM layer


def buildModel0(sequence_length, vocabSz, auxFeatureCount,
                LSTMinternalLayerSz, embedLayerDim,
                LSTMdropoutFrac=0.40,
                denseLayerDim=64,
                dropoutFrac=0.15,
                softMaxCt=16):

    """
    INPUTS:
    sequence_length		int, number of LSTM units
    vocabSz			int, size of vocabulary
    auxFeatureCount		int, count of auxiliary (side) features
    LSTMinternalLayerSz		int, size of layers within LSTM units
    embedLayerDim		int, dimension of embedding layer
    denseLayerDim		int, dimension of dense layers, default: 64
    softMaxCt			int, dimension of softmax output, default: 16
    dropoutFrac			int, dropout rate, default: 0.15
    LSTMdropoutFrac		int, dropout rate for LSTMs, default: 0.40
    """

    # Headline input: meant to receive sequences of *sequence_length* integers,
    #  between 1 and *vocabSz*.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(sequence_length,), dtype='int32',
                       name='MainInput')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 64-dimensional vectors.
    x = Embedding(output_dim=embedLayerDim, input_dim=vocabSz,
                  input_length=sequence_length, trainable=True,
                  name='EmbedLayer')(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = Bidirectional(LSTM(LSTMinternalLayerSz,
                                  dropout=dropoutFrac,
                                  recurrent_dropout=LSTMdropoutFrac,
                                  return_sequences=True,
                                  name='BidirectionalLSTM'))(x)
    print(f"lstm_out.shape: {lstm_out.shape}")

    auxiliary_input = Input(shape=(auxFeatureCount,), name='NumericalFeatures')
    x = concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(denseLayerDim, activation='relu')(x)
    x = Dense(denseLayerDim, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(denseLayerDim, activation='softmax',
                        name='SoftmaxOutput')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)

    return model


# This version adds a second LSTM layer
def buildModel1(sequence_length, vocabSz, auxFeatureCount,
                LSTMinternalLayerSz, embedLayerDim,
                LSTMdropoutFrac=0.40,
                denseLayerDim=64,
                dropoutFrac=0.15,
                softMaxCt=16):

    """
    INPUTS:
    sequence_length		int, number of LSTM units
    vocabSz			int, size of vocabulary
    auxFeatureCount		int, count of auxiliary (side) features
    LSTMinternalLayerSz		int, size of layers within LSTM units
    embedLayerDim		int, dimension of embedding layer
    denseLayerDim		int, dimension of dense layers, default: 64
    softMaxCt			int, dimension of softmax output, default: 16
    dropoutFrac			int, dropout rate, default: 0.15
    LSTMdropoutFrac		int, dropout rate for LSTMs, default: 0.40
    """

    # Headline input: meant to receive sequences of *sequence_length* integers,
    #  between 1 and *vocabSz*.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(sequence_length,), dtype='int32',
                       name='MainInput')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 64-dimensional vectors.
    x = Embedding(output_dim=embedLayerDim, input_dim=vocabSz,
                  input_length=sequence_length, trainable=True,
                  name='EmbedLayer')(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out_1 = Bidirectional(LSTM(LSTMinternalLayerSz,
                                    dropout=dropoutFrac,
                                    recurrent_dropout=LSTMdropoutFrac,
                                    return_sequences=True,
                                    name='BidirectionalLSTM'))(x)
    print(f"lstm_out_1.shape: {lstm_out_1.shape}")
    lstm_out = LSTM(LSTMinternalLayerSz,
                    dropout=dropoutFrac,
                    recurrent_dropout=LSTMdropoutFrac,
                    name='LSTM2')(lstm_out_1)
    print(f"lstm_out.shape: {lstm_out.shape}")

    auxiliary_input = Input(shape=(auxFeatureCount,), name='NumericalFeatures')
    x = concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(denseLayerDim, activation='relu')(x)
    x = Dense(denseLayerDim, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(denseLayerDim, activation='softmax',
                        name='SoftmaxOutput')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)

    return model


if __name__ == "__main__":

    # ## Model Parameters

    # * testFrac		fraction of data set withheld
    # * maxVocabCt		vocabulary size to be returned by Tokenizer
    # * LSTMlayerUnits		# units within each activation unit in LSTMs]
    # * embeddingDim		size of dimension for generated embeddings
    # * auxFeaturesCt		# of features in auxiliary data
    # * classCt			# classes (softmax output dim)
    # * dropoutFrac		dropout fraction
    # * LSTMdropoutFrac		dropout fraction within LSTMs
    # * batchSz			size of batches
    # * epochCt			number of epochs to run

    testFrac = 0.4
    maxVocabCt = 80000
    maxCommentWords = 60		# 40|60
    LSTMlayerUnits = 64
    embeddingDim = 64
    auxFeaturesCt = 1
    classCt = 56
    dropoutFrac = 0.20			# 0.20|0.30
    LSTMdropoutFrac = 0.50		# 0.35|0.50
    batchSz = 64
    epochCt = 10
    denseLayerDim = 64

    classColumn = 'fda_q_fixed'
    useModel = 1

    tensorBoardLogDir = './tensorBoardLogs'
    os.makedirs(tensorBoardLogDir, exist_ok=True)

    modelInstanceDir = \
        (f"{useModel}vocabCt{maxVocabCt:06d}"
         + f"maxCommentLen{maxCommentWords:03d}"
         + f"auxFeaturesCt{auxFeaturesCt:02d}"
         + f"classCt{classCt:02d}"
         + f"embedDim{embeddingDim:03d}"
         + f"LSTMlayerSz{LSTMlayerUnits:03d}batchSz{batchSz:03d}"
         + f"dropoutFrac{dropoutFrac:4.2f}"
         + f"LSTMdropoutFrac{LSTMdropoutFrac:4.2f}")

    # Redirect stdout to modelInstanceDir/stdout:

    os.makedirs(tensorBoardLogDir + '/' + modelInstanceDir, exist_ok=True)
    stdoutFile = os.path.join(tensorBoardLogDir, modelInstanceDir, 'stdout')
    sys.stdout = open(stdoutFile, 'w')

    # Redirect stderr to modelInstanceDir/stderr:
    stderrFile = os.path.join(tensorBoardLogDir, modelInstanceDir, 'stderr')
    sys.stderr = open(stderrFile, 'w')

    fname = "SelectedInspectionReportData.rds"

    t0 = time()
    result = pyreadr.read_r(fname)
    df = result[None]
    df.fda_q_fixed = df.fda_q_fixed.astype('int')
    df.FDAISCRITICAL = df.FDAISCRITICAL.astype('int')
    Δt = time() - t0
    print(f"\n\nTime to read data Δt: {Δt: 4.1f}s.")

    print("df.shape: ", df.shape)

    print("df.head(6):\n", df.head(6).T.to_markdown())
    print("df.tail(6):\n", df.tail(6).T.to_markdown())

    # ### Basic summary

    print(df.info())

    print(df.describe().to_markdown())

    # ### Remove columns not needed
    df = df[['fda_q_fixed', 'VIOCOMMENT', 'FDAISCRITICAL']]
    print(df.info())

    classCts = df.fda_q_fixed.value_counts()
    print("classCts:\n", pd.DataFrame(classCts).to_markdown())

    # ### Word frequencies

    t0 = time()
    df['commentsWords'] = df['VIOCOMMENT'].apply(lambda s: s.split())
    t1 = time()
    Δt = t1 - t0
    print(f"Time to split words Δt: {Δt % 60.0:4.1f}s.")

    comments = list(df['commentsWords'])
    print("First comment tokens:\n", comments[0])
    print("Last comment tokens:\n", comments[-1])

    # * Add length of each comment to DataFrame as `wordFreq` column

    wordLens = [len(wordList) for wordList in comments]
    df['commentLen'] = wordLens
    commentLenMode = df['commentLen'].mode().values[0]

    # wordCtSorted = sorted(wordLens)
    # print("smallest word counts:\n", wordCtSorted[:50])
    # print("largest word counts:\n", wordCtSorted[-51:-1])

    # **Make space in memory**

    del wordLens
    # del wordCtSorted
    del df['commentsWords']

    # ### What FDA codes correspond to those comments having `wordFreq== 0`?

    print("\ndf.shape: ", df.shape)
    print(df[df['commentLen'] == 0].to_markdown())

    # **Can safely remove a couple of records from the 2nd-most populated
    # category**

    # * Originally there were 1307986 records in `df`, out of which 122314
    #   were in Class 49

    df = df[df['commentLen'] != 0]
    print("df.shape: ", df.shape)

    print(df.describe(percentiles=[0.01, 0.05, 0.15, 0.25, 0.5,
                                   0.75, 0.85, 0.95, 0.99]).to_markdown())
    FDAcodes = list(set(df['fda_q_fixed'].values))
    print(f"FDAcodes: {FDAcodes}")

    (dfTr, XcommentsTr, XauxTr, FDAcodesTr,
     dfTe, XcommentsTe, XauxTe,  FDAcodesTe, myRandomState) = \
        preprocessDatums(df, classColumn, testFrac)

    print("Back from preprocessDatums().")

    # ### Create the Model, compile it, run it

    np.random.seed(0)  # Set a random seed for reproducibility

    if useModel == 0:
        modelLSTM = buildModel0(maxCommentWords, maxVocabCt, auxFeaturesCt,
                                LSTMlayerUnits, embeddingDim,
                                denseLayerDim=denseLayerDim, softMaxCt=classCt)
    elif useModel == 1:
        modelLSTM = buildModel1(maxCommentWords, maxVocabCt, auxFeaturesCt,
                                LSTMlayerUnits, embeddingDim,
                                denseLayerDim=denseLayerDim, softMaxCt=classCt)
    else:
        raise ValueError(f"You've set useModel: {useModel}, but must be one "
                         + "of [0, 1].")

    print(modelLSTM.summary())

    # ### Compute weights for each class

    dfTr.fda_q_fixed.unique() - 1

    classWeights = \
        class_weight.compute_class_weight('balanced',
                                          dfTr.fda_q_fixed.unique() - 1,
                                          dfTr.fda_q_fixed - 1)
    print("classWeights:\n", classWeights)

    # ### Define callbacks for model checkpoints and TensorBoard

    checkpointDir = './checkpoints'

    checkpointPrefix = os.path.join(checkpointDir, modelInstanceDir,
                                    "ckpt{epoch:03d}")
    checkpointCallback = ModelCheckpoint(filepath=checkpointPrefix,
                                         save_weights_only=True)

    logsDir = os.path.join(tensorBoardLogDir, modelInstanceDir)

    os.makedirs(logsDir, exist_ok=True)
    tensorboardCallback = TensorBoard(log_dir=logsDir, histogram_freq=1)

    # ### Compile the model

    modelLSTM.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])

    # Place hardcopy plots in logsDir

    plotDir = logsDir
    print(f"\nplotDir: {plotDir}")

    # Create model graph plot
    plot_model(modelLSTM, to_file=os.path.join(plotDir, 'modelGraph.png'))

    # ### Run the model

    # verbose=2 setting prints 1 line per epoch, while verbose=1 prints a line
    # every second or so!
    history = modelLSTM.fit(batch_size=batchSz,
                            callbacks=[checkpointCallback,
                                       tensorboardCallback],
                            class_weight=classWeights,
                            epochs=epochCt,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=2,
                            x=[XcommentsTr, XauxTr],
                            y=FDAcodesTr)

    # history = modelLSTM.fit({'main_input': paddedSequences,
    #                          'numerical_input': bools},
    #                         {'main_output': FDAcodes},
    #                         epochs=5, batch_size=batchSz)

    modelLSTM.save(logsDir + '/save.h5')

    # ### Do inference on test

    softmaxOut = modelLSTM.predict(x=[XcommentsTe, XauxTe])

    yPred = np.argmax(softmaxOut, axis=1) + 1

    print(dfTe.head(3))
    print(f"dfTe.shape: {dfTe.shape}")

    # ### Overall accuracy, precision, recall

    yTe = dfTe.fda_q_fixed
    confusionMat = confusion_matrix(yTe, yPred)
    print("confusion matrix:\n", confusionMat)

    # np.where(np.sum(confusionMat, axis=0) == 0)

    accuracy = np.trace(confusionMat)/np.sum(confusionMat)
    recall = np.diag(confusionMat)/np.sum(confusionMat, axis=1)
    precision = np.diag(confusionMat)/np.sum(confusionMat, axis=0)
    print(f"accuracy: {accuracy:0.3f}, "
          f"<precision>: {np.mean(precision):0.3f}, "
          f"<recall>: {np.mean(recall):0.3f}")

    # ### Recall, precision by class

    print(metrics.classification_report(yTe, yPred,
                                        target_names=[str(c)
                                                      for c in FDAcodes]))

    classCts = dfTe['fda_q_fixed'].value_counts()

    recall = np.diag(confusionMat)/np.sum(confusionMat, axis=1)
    precision = np.diag(confusionMat)/np.sum(confusionMat, axis=0)
    f1 = 2.0*precision*recall/(precision + recall)
    print("class\tprecision\trecall\tf1\tsize")

    for FDAcode, classCt in classCts.iteritems():
        print(f"{FDAcode}\t{precision[FDAcode - 1]:0.3f}\t\t"
              + f"{recall[FDAcode - 1]:0.3f}\t{f1[FDAcode - 1]:0.3f}"
              + f"\t\t{classCt:d}")

    # ### Plot confusion matrices

    labelFontSz = 16
    tickFontSz = 13
    titleFontSz = 20

    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    ph.plotConfusionMatrix(confusionMat,
                           ax=ax,
                           dir=plotDir,
                           saveAs='pdf',
                           titleFontSz=titleFontSz,
                           titleText='LSTM',
                           xlabelFontSz=labelFontSz,
                           xlabels=FDAcodes,
                           xtickFontSz=tickFontSz,
                           ylabelFontSz=labelFontSz,
                           ylabels=FDAcodes,
                           ytickFontSz=tickFontSz)

    # ### Plot recall confusion matrix (normalized by row)
    # * diagonal elements now represent the *recall* for each class

    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    ph.plotConfusionMatrix(confusionMat,
                           ax=ax,
                           dir=plotDir,
                           saveAs='pdf',
                           titleFontSz=titleFontSz,
                           titleText='LSTM',
                           type='recall',
                           xlabelFontSz=labelFontSz,
                           xlabels=FDAcodes,
                           xtickFontSz=tickFontSz,
                           ylabelFontSz=labelFontSz,
                           ylabels=FDAcodes,
                           ytickFontSz=tickFontSz)

    # ### Plot precision confusion matrix (normalized by column)
    # * diagonal elements now represent the *precision* for each class

    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    ph.plotConfusionMatrix(confusionMat,
                           ax=ax,
                           dir=plotDir,
                           saveAs='pdf',
                           titleFontSz=titleFontSz,
                           titleText='LSTM',
                           type='precision',
                           xlabelFontSz=labelFontSz,
                           xlabels=FDAcodes,
                           xtickFontSz=tickFontSz,
                           ylabelFontSz=labelFontSz,
                           ylabels=FDAcodes,
                           ytickFontSz=tickFontSz)

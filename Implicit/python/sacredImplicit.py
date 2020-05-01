#!/usr/bin/python3
# coding: utf-8

# Using `Implicit` and Bayesian Personalized Ranking (BPR) for Recommendations

# This follow's Ben Frederickson's [Finding Similar Music using Matrix
#  Factorization](https://www.benfrederickson.com/matrix-factorization/).

# ### Python imports

import sys

import argparse
import codecs
import logging
import time
import os
from time import time, asctime, localtime

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import itertools
import copy
from sklearn.metrics import mean_squared_error

# sys.path.append('/usr/local/lib/python3.6/dist-packages/implicit-0.4.0-py3.6-linux-x86_64.egg')
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares,
                                      FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.evaluation import (precision_at_k, mean_average_precision_at_k,
                                 ndcg_at_k, AUC_at_k, train_test_split)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.datasets.movielens import get_movielens
from implicit.datasets.reddit import get_reddit
from implicit.datasets.sketchfab import get_sketchfab
from implicit.datasets.million_song_dataset import get_msd_taste_profile
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)


import tqdm

from sacred import Experiment, host_info_gatherer
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

# ### Global dicts

models = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "lmf": LogisticMatrixFactorization,
          "bm25": BM25Recommender}

dataSets = {"lastfm": get_lastfm,
            "movielens": get_movielens,
            "reddit": get_reddit,
            "sketchfab": get_sketchfab,
            "million_song": get_msd_taste_profile}
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ## Functions

# ### `trainTestSplit()`

# def trainTestSplit(ratings, splitCount, fraction=None):
#     """
#     Stolen from Ethan Rosenthal's Intro to Implicit Matrix Factorization:
#     Classic ALS with Sketchfab Models
#     https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

#     Split recommendation data into train and test sets

#     In order to track precision@{k} as an optimization metric, it's necessary
#     to only work with. A k of 5 would be nice. However, if I move 5 items
#     from training to test for some of the users, then they may not have any
#     data left in the training set (remember they had a minimum 5 likes).
#     Thus, the train_test_split only looks for people who have at least 2*k
#     (10 in this case) likes before moving some of their data to the test set.
#     This obviously biases our cross-validation towards users with more likes.
#     So it goes.

#     Params
#     ------
#     ratings : scipy.sparse matrix
#         Interactions between users and items.
#     splitCount : int
#         Number of user-item-interactions per user to move
#         from training to test set.
#     fractions : float
#         Fraction of users to split off some of their
#         interactions into test set. If None, then all
#         users are considered.
#     """

#     # Note: likely not the fastest way to do things below.
#     train = ratings.copy().tocoo()
#     test = sparse.lil_matrix(train.shape)

#     if fraction:
#         try:
#             userIndex = np.random.choice(
#                 np.where(np.bincount(train.row) >= splitCount*2)[0],
#                 replace=False,
#                 size=np.int32(np.floor(fraction*train.shape[0]))
#             ).tolist()
#         except ValueError:
#             print(f"Not enough users with > {2*splitCount} "
#                   f"interactions to obtain a fraction of {fraction}.")
#         print('Try succeeded!')
#     else:
#         userIndex = range(train.shape[0])

#     train = train.tolil()

#     for user in userIndex:
#         testRatings = np.random.choice(ratings.getrow(user).indices,
#                                        size=splitCount,
#                                        replace=False)
#         train[user, testRatings] = 0.0
#         # These are just 1.0 right now
#         test[user, testRatings] = ratings[user, testRatings]

#     # Test and training are truly disjoint
#     assert(train.multiply(test).nnz == 0)
#     return train.tocsr(), test.tocsr(), userIndex

# ### `calculateMSE()`

# def calculateMSE(model, ratings, userIndex=None):
#     preds = model.predict_for_customers()
#     if userIndex:
#         return mean_squared_error(ratings[userIndex, :].toarray().ravel(),
#                                   preds[userIndex, :].ravel())

#     return mean_squared_error(ratings.toarray().ravel(),
#                               preds.ravel())

# ### `precisionAtK()`

# def precisionAtK(model, ratings, k=5, userIndex=None):
#     if not userIndex:
#         userIndex = range(ratings.shape[0])
#     ratings = ratings.tocsr()
#     precisions = []
#     # Note: line below may become infeasible for large datasets.
#     predictions = model.predict_for_customers()
#     for user in userIndex:
#         # In case of large dataset, compute predictions row-by-row like below
#         # predictions = np.array([model.predict(row, i) for i in
#                                  xrange(ratings.shape[1])])
#         topK = np.argsort(-predictions[user, :])[:k]
#         labels = ratings.getrow(user).indices
#         precision = float(len(set(topK) & set(labels))) / float(k)
#         precisions.append(precision)
#     return np.mean(precisions)


def calculateSimilarArtists(outputFilename, dataset, modelName="als"):
    """
    Generates a list of similar artists in lastfm by utilizing the
    'similar_items' api of the models
    """

    artists, users, plays = fetchDataset(dataset, volubility=2)
    model = getModel(modelName, volubility=2)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", modelName)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", modelName,
                  time.time() - start)

    # write out similar artists by popularity
    start = time.time()
    logging.debug("calculating top artists")

    user_count = np.ediff1d(plays.indptr)
    to_generate = sorted(np.arange(len(artists)), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    logging.debug("writing similar items")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(outputFilename, "w", "utf8") as o:
            for artistid in to_generate:
                artist = artists[artistid]
                for other, score in model.similar_items(artistid, 11):
                    o.write("%s\t%s\t%s\n" % (artist, artists[other], score))
                progress.update(1)

    logging.debug("generated similar artists in %0.2fs",  time.time() - start)


def calculateRecommendations(outputFilename, modelName="als"):
    """
    Generates artist recommendations for each user in the dataset
    """

    artists, users, plays = fetchDataset(dataset, volubility=2)
    model = getModel(modelName, volubility=2)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", modelName)
    start = time.time()
    model.fit(plays)
    logging.debug(f"trained model '{modelName}' in "
                  f"{time.time() - start:0.2fs}")

    # generate recommendations for each user and write out to a file
    start = time.time()
    user_plays = plays.T.tocsr()
    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(outputFilename, "w", "utf8") as o:
            for userid, username in enumerate(users):
                for artistid, score in model.recommend(userid, user_plays):
                    o.write("%s\t%s\t%s\n" % (username, artists[artistid],
                                              score))
                progress.update(1)
    logging.debug("generated recommendations in %0.2fs",  time.time() - start)


def getModel(modelName, volubility=1, params=None):
    """
    Instantiates a model class, using provided params, or defaults

    INPUTS:
        modelName	str, one of ['als', 'nmslib_als', 'annoy_als',
                        'faiss_als', 'tfidf', 'cosine', 'bpr', 'lmf', 'bm25']
        params		dict, "suitable" key-value pairs for model
                        description, training conditions, etc.

    The models:

    als:	Alternating LeastSquares
    nmslib_als:	NMS lib Alternating LeastSquares
    annoy_als:	Annoy Alternating LeastSquares
    faiss_als:	Faiss Alternating LeastSquares
    tfidf:	TF-IDF recommender
    cosine:	Cosine recommender
    bpr:	Bayesian personalized ranking
    lmf:	Logistic matrix factorization
    bm25:	BM-25 based recommender


    """
    if volubility > 0:
        print("getting model %s" % modelName)

    modelClass = models.get(modelName)
    if not modelClass:
        raise ValueError("Unknown Model '%s'" % modelName)

    # some default params
    if params:
        if issubclass(modelClass, AlternatingLeastSquares):
            params['dtype'] = np.float32
        elif modelName == "bpr":
            params['dtype'] = np.float32
            # params['verify_negative_samples'] = True
    else:
        if issubclass(modelClass, AlternatingLeastSquares):
            params = {'factors': 32, 'dtype': np.float32}
        elif modelName == "bm25":
            params = {'K1': 100, 'B': 0.5}
        elif modelName == "bpr":
            params = {'factors': 63, 'dtype': np.float32,
                      'verify_negative_samples': True}
        elif modelName == "lmf":
            params = {'factors': 30, "iterations": 40, "regularization": 1.5}
        else:
            params = {}

    if volubility > 1:
        print(modelName.title)

    return modelClass(**params)


def fetchDataset(dataset, volubility=1):
    """
    If not already in cache directory, /data1/mark/implicit_datasets,
    fetches a data set, storing copy in cache.

    INPUT:
        dataset		str, one of ['lastfm', 'movielens', 'reddit',
                        'sketchfab', 'million_song']

    """

    if volubility > 0:
        print(f"getting dataset {dataset}")
    getdata = dataSets.get(dataset)

    if not getdata:
        raise ValueError(f"Unknown Model {dataset}")
    artists, users, plays = getdata()

    if volubility > 1:
        print(f"type(artists): {type(artists)}")
        print(f"type(users): {type(users)}")
        print(f"type(plays): {type(plays)}")

    return artists, users, plays


def printLog(row, header=False, spacing=12, outFile=None):
    if outFile is None:
        outFile = sys.stdout
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, float):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        outFile.write(top + "\n")
        outFile.write(middle + "\n")
        outFile.write(bottom + "\n")
        outFile.flush()
    else:
        outFile.write(middle + "\n")
        outFile.write(top + "\n")
        outFile.flush()


def learningCurve(model, train, test, epochs, outFile=None,
                  k=5, showProgress=True, numThreads=12):
    # if not userIndex:
    #     userIndex = range(train.shape[0])
    prevEpoch = 0

    pAtK = []
    MAPatK = []
    NDCGatK = []
    AUCatK = []

    headers = ["epochs", f"p@{k}", f"MAP@{k}", f"NDCG@{k}", f"AUC@{k}"]
    printLog(headers, header=True, outFile=outFile)

    for epoch in epochs:
        model.iterations = epoch - prevEpoch
        if not hasattr(model, "user_vectors"):
            model.fit(train, show_progress=showProgress)
        else:
            model.fit_partial(train, show_progress=showProgress)
        pAtK.append(precision_at_k(model, train.T.tocsr(), test.T.tocsr(),
                                   K=k, show_progress=showProgress,
                                   num_threads=numThreads))
        MAPatK.append(mean_average_precision_at_k(model, train.T.tocsr(),
                                                  test.T.tocsr(), K=k,
                                                  show_progress=showProgress,
                                                  num_threads=numThreads))
        NDCGatK.append(ndcg_at_k(model, train.T.tocsr(), test.T.tocsr(),
                                 K=k, show_progress=showProgress,
                                 num_threads=numThreads))
        AUCatK.append(AUC_at_k(model, train.T.tocsr(), test.T.tocsr(),
                               K=k, show_progress=showProgress,
                               num_threads=numThreads))
        row = [epoch, pAtK[-1], MAPatK[-1], NDCGatK[-1], AUCatK[-1]]
        printLog(row, outFile=outFile)
        prevEpoch = epoch

    return model, pAtK, MAPatK, NDCGatK, AUCatK


def gridSearchLearningCurve(modelName, train, test, paramGrid, numThreads=12,
                            k=5, showProgress=True, epochs=range(2, 10, 2),
                            LCfile='../LearningCurves.txt'):
    """
    "Inspired" (stolen) from sklearn gridsearch
    https://github.com/scikit-learn/scikit-learn/blob/master/..
    ../sklearn/model_selection/_search.py
    """

    curves = []
    keys, values = zip(*paramGrid.items())

    with open(LCfile, 'w') as outFile:
        for val in itertools.product(*values):
            params = dict(zip(keys, val))
            thisModel = getModel(modelName, volubility=2)
            outFile.write(str(type(thisModel)) + "\n")
            outFile.flush()

            printLine = []
            for key, value in params.items():
                setattr(thisModel, key, value)
                printLine.append((key, value))

            outFile.write(' | '.join(f'{key}: {value}'
                                     for (key, value) in printLine) + "\n")
            outFile.flush()

            _, pAtK, MAPatK, NDCGatK, AUCatK = \
                learningCurve(thisModel, train, test, epochs, k=k,
                              outFile=outFile, showProgress=showProgress,
                              numThreads=numThreads)

            curves.append({'params': params,
                           f"p@{k}": pAtK, f"MAP@{k}": MAPatK,
                           f"NDCG@{k}": NDCGatK, f"AUC@{k}": AUCatK})
            del thisModel

    return curves


@host_info_gatherer('host IP address')
def ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


ex = Experiment('Implicit models testing',
                additional_host_info=[ip])


@ex.config
def cfg():
    """
    # Default configuration values:
    'model': 'als'
    'dataset': 'lastfm',
    'factors': 64,
    'k': 6,
    'λ': 0.01,
    'α': 40.0,
    'iterations': 15,
    'progressBar': True,
    'useGPU': True,
    'threadCt': 0
    """

    # Default configuration values:
    modelName = 'als'
    datasetName = 'lastfm'
    factorCt = 64
    k = 6
    λ = 0.01
    α = 40.0
    maxIters = 15
    showProgress = False
    useGPU = True
    threadCt = 0


@ex.automain
def run(modelName, datasetName, factorCt, k, λ, α,
        maxIters, showProgress, useGPU, threadCt):

    if modelName == 'als':
        model = getModel(modelName, volubility=2,
                         params={'factors': factorCt,
                                 'regularization': λ,
                                 'iterations': maxIters,
                                 'use_gpu': useGPU})
    else:
        model = getModel(modelName, volubility=2,
                         params={'factors': factorCt,
                                 'regularization': λ,
                                 'alpha': α,
                                 'iterations': maxIters,
                                 'use_gpu': useGPU})

    artists, users, plays = fetchDataset(datasetName, volubility=2)

    print(artists.shape, users.shape, plays.shape, flush=True)

    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        print("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # print(asctime(localtime()))
    # t0 = time()
    plays = plays.tocsr()
    # print(f"Δt: {time() - t0:5.1f}s")

    train, test = train_test_split(plays, train_percentage=0.8)

    print("Training model")
    print(asctime(localtime()), flush=True)
    t0 = time()

    model.fit(train, show_progress=showProgress)
    print(f"Δt: {time() - t0:5.1f}s", flush=True)

    trainTscr = train.T.tocsr()
    testTscr = test.T.tocsr()

    print(f"Computing p@{k} ...", flush=True)
    t0 = time()
    pAtK = precision_at_k(model, trainTscr, testTscr, K=k,
                          show_progress=showProgress,
                          num_threads=threadCt)
    ex.log_scalar(f"p@{k}", pAtK)
    print(f"Δt: {time() - t0:5.1f}s")
    print(f"Computing MAP@{k} ...", flush=True)
    t0 = time()
    MAPatK = mean_average_precision_at_k(model, trainTscr, testTscr, K=k,
                                         show_progress=showProgress,
                                         num_threads=threadCt)
    ex.log_scalar(f"MAP@{k}", MAPatK)
    print(f"Δt: {time() - t0:5.1f}s")
    print(f"Computing NDCG@{k} ...", flush=True)
    t0 = time()
    NDCGatK = ndcg_at_k(model, trainTscr, testTscr, K=k,
                        show_progress=showProgress,
                        num_threads=threadCt)
    ex.log_scalar(f"NDCG@{k}", NDCGatK)
    AUCatK = AUC_at_k(model, trainTscr, testTscr, K=k,
                      show_progress=showProgress,
                      num_threads=threadCt)
    ex.log_scalar(f"AUC@{k}", AUCatK)
    print(f"Δt: {time() - t0:5.1f}s")
    print(f"p@{k}: {pAtK:6.4f}, MAP@{k}: {MAPatK:6.4f}"
          f"NDCG@{k}: {NDCGatK:6.4f}, AUC@{k}: {AUCatK:6.4f}", flush=True)

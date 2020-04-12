#!/usr/bin/python3

""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/

This code will automically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html.
"""
import argparse
import codecs
import logging
import time
import sys

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares,
                                      FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.datasets.movielens import get_movielens
from implicit.datasets.reddit import get_reddit
from implicit.datasets.sketchfab import get_sketchfab
from implicit.datasets.million_song_dataset import get_msd_taste_profile
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

# maps command line model argument to class name
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


def getModel(modelName):
    print("getting model %s" % modelName)
    model_class = models.get(modelName)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % modelName)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 16, 'dtype': np.float32}
    elif modelName == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif modelName == "bpr":
        params = {'factors': 63}
    elif modelName == "lmf":
        params = {'factors': 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def calculateSimilarArtists(output_filename, dataset, modelName="als"):
    """
    Generates a list of similar artists in lastfm by utilizing the
    'similar_items' api of the models
    """

    print(f"getting dataset {dataset}")
    # artists, users, plays = get_lastfm()
    getdata = dataSets.get(dataset)
    if not getdata:
        raise ValueError(f"Unknown Model {dataset}")
    artists, users, plays = getdata()
    # sys.exit()

    # create a model from the input data
    model = getModel(modelName)

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
        with codecs.open(output_filename, "w", "utf8") as o:
            for artistid in to_generate:
                artist = artists[artistid]
                for other, score in model.similar_items(artistid, 11):
                    o.write("%s\t%s\t%s\n" % (artist, artists[other], score))
                progress.update(1)

    logging.debug("generated similar artists in %0.2fs",  time.time() - start)


def calculateRecommendations(output_filename, modelName="als"):
    """ Generates artist recommendations for each user in the dataset """
    # train the model based off input params
    print(f"getting dataset {dataset}")

    # artists, users, plays = get_lastfm()
    getdata = dataSets.get(dataset)
    if not getdata:
        raise ValueError(f"Unknown dataset: {dataset}.")
    # sys.exit()

    artists, users, plays = getdata()
    print(type(artists))
    print(type(users))
    print(type(plays))
    sys.exit(1)

    # create a model from the input data
    model = getModel(modelName)

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
        with codecs.open(output_filename, "w", "utf8") as o:
            for userid, username in enumerate(users):
                for artistid, score in model.recommend(userid, user_plays):
                    o.write("%s\t%s\t%s\n" % (username, artists[artistid],
                                              score))
                progress.update(1)
    logging.debug("generated recommendations in %0.2fs",  time.time() - start)


def trainTestSplit(ratings, splitCount, fraction=None):
    """
    Stolen from Ethan Rosenthal's Intro to Implicit Matrix Factorization:
    Classic ALS with Sketchfab Models
    https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

    Split recommendation data into train and test sets

    Params
    ------
    ratings : scipy.sparse matrix
        Interactions between users and items.
    splitCount : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """

    # Note: likely not the fastest way to do things below.
    train = ratings.copy().tocoo()
    test = sparse.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= splitCount * 2)[0],
                replace=False,
                size=np.int32(np.floor(fraction * train.shape[0]))
            ).tolist()
        except ValueError:
            print(f"Not enough users with > {2*splitCount} "
                  f"interactions for fraction of {fraction}")
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices,
                                        size=splitCount,
                                        replace=False)
        train[user, test_ratings] = 0.
        # These are just 1.0 right now
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index


if __name__ == "__main__":
    myDescription = ("Generates similar artists on the last.fm dataset or "
                     "generates personalized recommendations for each user.")
    parser = \
        argparse.ArgumentParser(description=myDescription,
                                formatter_class=argparse
                                .ArgumentDefaultsHelpFormatter)

    helpStr = 'Output file name. (Omit to go with parameter-based naming)'
    parser.add_argument('--output-base', type=str,  # default='similar-artists'
                        dest='outputfile', help=helpStr)
    helpStr = f"model to calculate ({', '.join(models.keys())})"
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help=helpStr)
    helpStr = f"dataset ({', '.join(dataSets.keys())})"
    parser.add_argument('--dataset', type=str, default='lastfm',
                        dest='dataset', help=helpStr)
    helpStr = ("Recommend items for each user rather than calculate "
               "similar_items")
    parser.add_argument('--recommend',
                        help=helpStr,
                        action="store_true")
    helpStr = "Parameters to pass to the model, formatted as 'KEY=VALUE"
    parser.add_argument('--param', action='append',
                        help=helpStr)

    args = parser.parse_args()
    print(args)

    if args.outputfile:
        outFile = args.outputfile
    elif args.recommend:
        outFile = f"recommend-{args.model}-{args.dataset}.tsv"
    else:
        outFile = f"similarItems-{args.model}-{args.dataset}.tsv"
    print(f"Writing output to {outFile}")

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculateRecommendations(outFile, args.dataset,
                                 modelName=args.model)
    else:
        calculateSimilarArtists(outFile, args.dataset,
                                modelName=args.model)

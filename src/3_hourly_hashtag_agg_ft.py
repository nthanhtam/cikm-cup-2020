#!/usr/bin/env python
# coding: utf-8

__author__ = "Sheng Jie Lui"
__email__ = "shengjie@aidatech.io"
__contributors__ = ["Sheng Jie Lui", "Tam Nguyen"]


import re
import os
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm_notebook as tqdm
import gc
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import FastText
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import gensim.models.doc2vec as doc

from sklearn.neighbors import NearestNeighbors
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def train_doc2_vec(
    data_path,
    embed_size=8,
    epoch=10,
    min_count=1,
    window=4,
    workers=8,
    model_path="./models/doc2vec_hashtags_hourly.model",
    train_feature_file="",
    valid_feature_file="",
    test_feature_file="",
):

    df_all = pd.read_csv(data_path)
    logging.info("Build Training Corpus for Doc2Vec")
    start = time.time()
    hashtags = df_all["processed_hashtags"].tolist()
    train_corpus = [
        doc.TaggedDocument(word, [idx]) for idx, word in enumerate(hashtags)
    ]
    logging.info("Train Doc2Vec model")
    model = doc.Doc2Vec(
        vector_size=embed_size,
        min_count=min_count,
        epochs=epoch,
        window=window,
        workers=workers,
    )
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))
    model.save(model_path)

    df_all = df_all[["date","hour"]]
    feat = []
    for i in range(df_all.shape[0]):
        feat.append(model.docvecs[i])

    features = ["hashtags_d2v_hourly_%i" % x for x in range(len(feat[0]))]
    feat = pd.DataFrame(feat, columns=features)
    df_all = pd.concat((df_all, feat), axis=1)
    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2

    df = pd.read_csv('./features/date_cols.csv.gz')
    df = df.merge(df_all, how='left', on=['date','hour'])

    train = df[df["split"] == TRN_SET]
    validation = df[df["split"] == VAL_SET]
    test = df[df["split"] == TST_SET]

    logging.info("saving training features")
    train[features].to_csv(train_feature_file, index=False, compression="gzip")
    logging.info("saving validation features")
    validation[features].to_csv(valid_feature_file, index=False, compression="gzip")
    logging.info("saving testing featured")
    test[features].to_csv(test_feature_file, index=False, compression="gzip")


def main(
    data_path,
    embed_size,
    epoch,
    train_feature_file,
    valid_feature_file,
    test_feature_file,
):

    train_doc2_vec(
        data_path=os.path.join(data_path, "hourly_hashtags.csv.gz"),
        model_path="./models/doc2vec_hashtags_hourly.model",
        embed_size=embed_size,
        epoch=epoch,
        train_feature_file=train_feature_file,
        valid_feature_file=valid_feature_file,
        test_feature_file=test_feature_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, dest="data_path")
    parser.add_argument("--model-path", required=False, dest="model_path")
    parser.add_argument("--out-path", required=False, dest="out_path")
    parser.add_argument(
        "--embed-size", default=64, type=int, required=False, dest="embed_size"
    )
    parser.add_argument("--epoch", required=False, default=10, type=int, dest="epoch")
    parser.add_argument(
        "--train-feature-file", required=True, dest="train_feature_file"
    )
    parser.add_argument(
        "--valid-feature-file", required=True, dest="valid_feature_file"
    )
    parser.add_argument("--test-feature-file", required=True, dest="test_feature_file")

    args = parser.parse_args()
    start = time.time()
    main(
        args.data_path,
        args.embed_size,
        args.epoch,
        args.train_feature_file,
        args.valid_feature_file,
        args.test_feature_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

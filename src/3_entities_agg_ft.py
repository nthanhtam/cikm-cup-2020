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


def preprocess(words):
    # Remove stop words
    stopwords = text.ENGLISH_STOP_WORDS
    out = " ".join([word for word in words.split() if word not in stopwords])
    return out


def extract_doc2vec_feat(words_list, model):
    words_vect = []
    for words in words_list:
        vect = model.infer_vector([words])
        words_vect.append(vect)
    return words_vect


def extract_entities(entities_list, pos=0):
    final_list = []
    for entities in entities_list:
        entity_list = entities.split(";")
        select_list = []
        for entity in entity_list:
            if ":" in entity:
                raw, annotated, score = entity.split(":")
                if pos == 0:
                    select = raw
                elif pos == 1:
                    select = annotated
                elif pos == 2:
                    select = score
                select_list.append(select)
        if len(select_list) <= 0:
            select_str = "null"
        else:
            select_str = " ".join(select_list)
            select_str = preprocess(select_str)
        final_list.append(select_str)
    return final_list


def get_username_agg_entities_p1(
    data_path,
    train_filename="train.csv",
    val_filename="validation.csv",
    test_filename="test.csv",
    outpath="username_agg_raw_entities.csv",
):

    df_train = pd.read_csv(os.path.join(data_path, train_filename))
    logging.info(f"Load Training Data: {df_train.shape}")
    df_val = pd.read_csv(os.path.join(data_path, val_filename))
    logging.info(f"Load Validation Data: {df_val.shape}")
    df_test = pd.read_csv(os.path.join(data_path, test_filename))
    logging.info(f"Load Testing Data: {df_test.shape}")
    df_all = pd.concat([df_train, df_val, df_test])
    logging.info(f"Concat All Data: {df_all.shape}")
    # Delete DataFrame to save memory
    del df_train
    del df_val
    del df_test

    logging.info(f"Extracting entities...")
    df_all["ex_entities"] = extract_entities(df_all["entities"].tolist(), pos=0)
    logging.info(f"Grouping by username...")
    df_agg = df_all.groupby("username")["ex_entities"].apply(",".join).reset_index()

    df_agg.to_csv(os.path.join(data_path, outpath), index=False)


def get_username_agg_entities_p2(
    data_path,
    model_path="word2vec_entities_raw_agg_user.model",
    outpath="username_agg_raw_entities_feat.csv.gz",
):
    df_agg = pd.read_csv(data_path)
    df_agg.fillna("null", inplace=True)
    df_agg = df_agg[["username", "ex_entities"]]
    model = doc.Doc2Vec.load(model_path)

    ex_entities_vect = extract_doc2vec_feat(
        words_list=df_agg["ex_entities"].tolist(), model=model
    )
    ex_entities_vect = np.array(ex_entities_vect)

    df_ex_entities_vect = pd.DataFrame(ex_entities_vect)
    df_ex_entities_vect.columns = [
        f"doc2vec_ft_{x}" for x in range(df_ex_entities_vect.shape[1])
    ]

    df_ex_entities_vect["username"] = df_agg["username"]
    doc2vec_ft_cols = [f"doc2vec_ft_{x}" for x in range(64)]
    select_cols = ["username"] + doc2vec_ft_cols

    df_ex_entities_vect = df_ex_entities_vect[select_cols]

    df_ex_entities_vect.to_csv(outpath, index=False, compression="gzip")


def train_doc2_vec(
    data_path,
    embed_size=64,
    epoch=10,
    min_count=1,
    window=4,
    workers=8,
    model_path="word2vec_entities_raw_agg_user.model",
):

    df_all = pd.read_csv(data_path)
    df_all.fillna("null", inplace=True)
    logging.info(f"Load All Data: {df_all.shape}")
    logging.info(f"Build Training Corpus for Doc2Vec")
    start = time.time()
    ex_entities = df_all["ex_entities"].tolist()
    train_corpus = [
        doc.TaggedDocument(word, [idx]) for idx, word in enumerate(ex_entities)
    ]
    logging.info(f"Train Doc2Vec model")
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


def main(data_path, embed_size, epoch):
    get_username_agg_entities_p1(
        data_path,
        train_filename="train.csv",
        val_filename="validation.csv",
        test_filename="test.csv",
        outpath="username_agg_raw_entities.csv",
    )

    train_doc2_vec(
        data_path=os.path.join(data_path, "username_agg_raw_entities.csv"),
        model_path="word2vec_entities_raw_agg_user.model",
        embed_size=embed_size,
        epoch=epoch,
    )

    get_username_agg_entities_p2(
        data_path=os.path.join(data_path, "username_agg_raw_entities.csv"),
        model_path="word2vec_entities_raw_agg_user.model",
        outpath="username_agg_raw_entities_feat.csv.gz",
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

    args = parser.parse_args()
    start = time.time()
    main(args.data_path, args.embed_size, args.epoch)

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

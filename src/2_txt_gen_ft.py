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
from sklearn.neighbors import NearestNeighbors
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def post_process(df, col_name):
    df[col_name].fillna("null", inplace=True)
    df.loc[
        (df[col_name].str.contains(pat="^[0-9 ]+$", regex=True) == True), col_name
    ] = "null"
    return df


def get_sentence_embedding(df_in, colname, embed_size=100):
    df = df_in.loc[(df_in[colname] != "null")][["tweet_id", colname]]
    df.reset_index(drop=True, inplace=True)
    col_values = df[colname].tolist()
    col_values_list = [x.split(" ") for x in col_values]

    # Build Word2Vec model
    model = Word2Vec(size=embed_size, min_count=1, window=4, workers=8)
    model.build_vocab(col_values_list)  # prepare the model vocabulary
    model.train(
        col_values_list, total_examples=model.corpus_count, epochs=model.iter
    )  # train word vectors

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(col_values)
    nonzero_row, nonzero_col = tfidf.nonzero()
    tf_idf_val = [tfidf[x, y] for x, y in zip(nonzero_row, nonzero_col)]

    vect_feat = vectorizer.get_feature_names()
    print(f"Length of vect_feat: {len(vect_feat)}")
    vect_feat_mapper = {}
    for idx, name in enumerate(vect_feat):
        vect_feat_mapper[idx] = name

    vocab_keyword = [vect_feat_mapper[x] for x in nonzero_col]
    word_embs = [model.wv[x] for x in vocab_keyword]

    df_word_embs = pd.DataFrame(word_embs)
    df_word_embs.columns = [f"ft_{colname}_{x}" for x in range(df_word_embs.shape[1])]
    print(f"df_word_embs: {df_word_embs.shape}")

    df_mapper = pd.DataFrame(
        {
            "doc_idx": nonzero_row,
            "vocab_idx": nonzero_col,
            "tfidf_val": tf_idf_val,
            "vocab_keyword": vocab_keyword,
        }
    )
    print(f"df_mapper: {df_mapper.shape}")
    df_mapper.reset_index(inplace=True)

    df_emb_ft = pd.concat([df_mapper, df_word_embs], axis=1)
    print(f"df_emb_ft: {df_emb_ft.shape}")

    df_emb_ft_tmp = df_emb_ft.copy()
    ft_cols = [x for x in df_emb_ft_tmp.columns if bool(re.search("^ft_", x))]

    # Multiply word Embedding with TF-IDF
    for idx, ft_col in enumerate(ft_cols):
        df_emb_ft_tmp[ft_col] = df_emb_ft_tmp[ft_col] * df_emb_ft_tmp["tfidf_val"]
        if idx % 10 == 0:
            print(f"{ft_col}: {idx}/100")

    mapper_dict = {
        k: "sum"
        for k in [
            x for x in df_emb_ft_tmp.columns if bool(re.search("^ft_|tfidf_val", x))
        ]
    }
    df_emb_ft_up = df_emb_ft_tmp.groupby("doc_idx").agg(mapper_dict)
    df_emb_ft_up.reset_index(inplace=True)

    for idx, ft_col in enumerate(ft_cols):
        df_emb_ft_up[ft_col] = df_emb_ft_up[ft_col] / df_emb_ft_up["tfidf_val"]
        if idx % 10 == 0:
            print(f"{ft_col}: {idx}/100")

    print(df_emb_ft_up.shape)
    df_merge = pd.concat([df, df_emb_ft_up], axis=1)
    print(df_merge.shape)

    df_all = df_in.copy()
    df_all.drop(columns=[colname], inplace=True)
    df_final = pd.merge(df_all, df_merge, how="left", on="tweet_id")

    for col in ft_cols:
        df_final[col] = df_final[col].fillna("0")

    return df_final


def main(
    data_path, train_feature_file, valid_feature_file, test_feature_file, embed_size
):
    outfile = "train_val_data_pre.csv.gz"
    df = pd.read_csv(os.path.join(data_path, outfile))
    logging.info(f"DataFrame has shape: {df.shape}")
    df["tweet_id"] = range(df.shape[0])
    logging.info(df.columns)

    df = post_process(df, "hashtags_pre")
    df = post_process(df, "mentions_pre")

    df = get_sentence_embedding(df, "hashtags_pre", embed_size)
    logging.info(df.columns)

    df = get_sentence_embedding(df, "mentions_pre", embed_size)

    df.drop(
        columns=[
            "hashtags_pre",
            "mentions_pre",
            "doc_idx_x",
            "tfidf_val_x",
            "doc_idx_y",
            "tfidf_val_y",
        ],
        inplace=True,
    )

    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2

    train = df[df["split"] == TRN_SET]
    validation = df[df["split"] == VAL_SET]
    test = df[df["split"] == TST_SET]

    sel_features = [x for x in train.columns if x.startswith("ft_")]

    logging.info("saving training features")
    train[sel_features].to_csv(train_feature_file, index=False, compression="gzip")
    logging.info("saving validation features")
    validation[sel_features].to_csv(valid_feature_file, index=False, compression="gzip")
    logging.info("saving testing featured")
    test[sel_features].to_csv(test_feature_file, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, dest="data_path")
    parser.add_argument(
        "--train-feature-file", required=True, dest="train_feature_file"
    )
    parser.add_argument(
        "--valid-feature-file", required=True, dest="valid_feature_file"
    )
    parser.add_argument("--test-feature-file", required=True, dest="test_feature_file")
    parser.add_argument(
        "--embed-size", required=False, default=100, type=int, dest="embed_size"
    )

    args = parser.parse_args()
    start = time.time()
    main(
        args.data_path,
        args.train_feature_file,
        args.valid_feature_file,
        args.test_feature_file,
        args.embed_size,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

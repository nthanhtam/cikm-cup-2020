#!/usr/bin/env python
# coding: utf-8

import re
import os
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm_notebook as tqdm
from urllib.parse import urlparse
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import FastText
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.neighbors import NearestNeighbors
from num2words import num2words
import argparse
import logging
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def load_data(filepath, feats):
    f = open(filepath, "r", encoding="utf-8")
    lines = f.readlines()
    df = pd.DataFrame(lines)
    df.columns = ["tmp_col"]

    df[feats] = df["tmp_col"].str.split("\t", expand=True,)
    df.drop(columns=["tmp_col"], inplace=True)
    return df


def preprocess_text(word):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    word = word.lower()
    # Remove emoji
    word = emoji_pattern.sub(r"", word)
    word = re.sub("[^A-Za-z0-9 ]+", "", word)
    word = word.replace("2019", "19")
    word = word.replace("coronavirus", "covid")

    return word


def preprocess_text2(word):
    if word == "":
        word = "null"
    if word.isdecimal():
        word = num2words(word)
    word = re.sub("[^A-Za-z0-9 ]+", "", word)
    return word


def preprocess_text3(word):
    if len(word) == 1:
        word = f"only{word}"
    word = re.sub("[^A-Za-z0-9 ]+", "", word)
    return word


def preprocess_text_agg(df, col_name):
    cols_raw = df[col_name].tolist()
    cols_pre = [preprocess_text(x) for x in cols_raw]
    cols_pre = [preprocess_text2(x) for x in cols_pre]
    final_words_list = []
    for col_pre in cols_pre:
        pre_word_list = []
        for word in col_pre.split(" "):
            pre_word = preprocess_text3(word)
            pre_word_list.append(pre_word)
        final_words = " ".join(pre_word_list)
        final_words_list.append(final_words)

    return final_words_list


def parse_entity(s):
    if s == "null;":
        return ""
    try:
        tokens = map(
            lambda x: x.split(":")[1].replace("%28", "").replace("%29", ""),
            s.split(";")[:-1],
        )
    except:
        tokens = s.split(";")

    return " ".join(tokens).lower()


def parse_url_netloc(s):
    if s.startswith("null;"):
        return ""
    result = [urlparse(y) for y in filter(lambda x: x.strip() != "", s.split(":-:"))]
    result = [x.netloc.replace(".", "_") for x in result]
    return " ".join(result).lower()


def parse_url_path(s):
    if s.startswith("null;"):
        return ""
    result = [urlparse(y) for y in filter(lambda x: x.strip() != "", s.split(":-:"))]
    result = [x.path for x in result]
    return " ".join(result)


def post_process(df, col_name):
    df[col_name].fillna("null", inplace=True)
    df.loc[
        (df[col_name].str.contains(pat="^[0-9 ]+$", regex=True) == True), col_name
    ] = "null"
    return df


def main(data_path):
    logging.info("loading data")
    outfile = "train_val_data_pre.csv.gz"
    df = pd.read_csv(os.path.join(data_path, outfile))
    logging.info(f"DataFrame has shape: {df.shape}")
    df["tweet_id"] = range(df.shape[0])
    logging.info(df.columns)

    logging.info("post-processing entity data")
    df = post_process(df, "hashtags_pre")
    df = post_process(df, "mentions_pre")

    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2

    train = df[df["split"] == TRN_SET]
    validation = df[df["split"] == VAL_SET]
    test = df[df["split"] == TST_SET]

    logging.info("process mention data")
    vocab_size = 1000000
    max_length = 50
    train_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in train["mentions_pre"]],
        maxlen=max_length,
        padding="pre",
    )
    valid_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in validation["mentions_pre"]],
        maxlen=max_length,
        padding="pre",
    )
    test_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in test["mentions_pre"]],
        maxlen=max_length,
        padding="pre",
    )

    logging.info("saving data")
    with open("features/mention_train_1m50.pkl", "wb") as f:
        pickle.dump(train_entity, f, protocol=4)
    with open("features/mention_valid_1m50.pkl", "wb") as f:
        pickle.dump(valid_entity, f, protocol=4)
    with open("features/mention_test_1m50.pkl", "wb") as f:
        pickle.dump(test_entity, f, protocol=4)

    logging.info("process hashtag data")
    vocab_size = 1000000
    max_length = 25
    train_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in train["hashtags_pre"]],
        maxlen=max_length,
        padding="pre",
    )
    valid_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in validation["hashtags_pre"]],
        maxlen=max_length,
        padding="pre",
    )
    test_entity = pad_sequences(
        [one_hot(x, vocab_size) for x in test["hashtags_pre"]],
        maxlen=max_length,
        padding="pre",
    )

    logging.info("saving data")
    with open("features/hashtag_train_1m25.pkl", "wb") as f:
        pickle.dump(train_entity, f, protocol=4)
    with open("features/hashtag_valid_1m25.pkl", "wb") as f:
        pickle.dump(valid_entity, f, protocol=4)
    with open("features/hashtag_test_1m25.pkl", "wb") as f:
        pickle.dump(test_entity, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, dest="data_path")

    args = parser.parse_args()
    start = time.time()
    main(args.data_path,)

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

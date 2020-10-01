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


def post_process(df, col_name):
    df[col_name].fillna("null", inplace=True)
    df.loc[
        (df[col_name].str.contains(pat="^[0-9 ]+$", regex=True) == True), col_name
    ] = "null"
    return df


def main(data_path, output_file="train_val_data_pre.csv.gz"):
    data_cols = "feature.name"
    datafile_01 = "train.data"
    datafile_02 = "validation.data"
    datafile_03 = "test.data"

    outfile = output_file

    logging.info("loading data")
    input_01_filepath = os.path.join(data_path, datafile_01)
    input_02_filepath = os.path.join(data_path, datafile_02)
    input_03_filepath = os.path.join(data_path, datafile_03)

    col_filepath = os.path.join(data_path, data_cols)
    feat = list(pd.read_csv(col_filepath, sep="\t").columns)

    df_train_raw = load_data(input_01_filepath, feat)
    print(f"Training data has shape: {df_train_raw.shape}")

    df_val_raw = load_data(input_02_filepath, feat)
    print(f"Validation data has shape: {df_val_raw.shape}")

    feat.remove("tweet_id")
    df_tst_raw = load_data(input_03_filepath, feat)
    print(f"Test data has shape: {df_tst_raw.shape}")

    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2
    df_train_raw["split"] = TRN_SET
    df_val_raw["split"] = VAL_SET
    df_tst_raw["split"] = TST_SET

    df_train_raw = df_train_raw[["split", "hashtags", "mentions"]]
    df_val_raw = df_val_raw[["split", "hashtags", "mentions"]]
    df_tst_raw = df_tst_raw[["split", "hashtags", "mentions"]]

    logging.info("preprocessing data")
    df_all = pd.concat([df_train_raw, df_val_raw, df_tst_raw])
    print(f"Train + Validation data has shape: {df_all.shape}")

    hashtags_pre = preprocess_text_agg(df_all, "hashtags")
    mentions_pre = preprocess_text_agg(df_all, "mentions")

    df_all["hashtags_pre"] = hashtags_pre
    df_all["mentions_pre"] = mentions_pre

    df_all = df_all[["split", "hashtags_pre", "mentions_pre"]]

    logging.info("saving data")
    df_all.to_csv(os.path.join(data_path, outfile), index=False, compression="gzip")
    print(f"Saved {os.path.join(data_path, outfile)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, dest="data_path")
    parser.add_argument("--output-file", required=True, dest="output_file")

    args = parser.parse_args()
    start = time.time()
    main(
        args.data_path, args.output_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "tam@aidatech.io"

import argparse
import logging
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def read_csv(df_file, feat, nrows=None):
    with open(df_file, "r", encoding="utf-8") as f:
        if nrows is None:
            lines = map(lambda x: x.strip(), f.readlines())
        else:
            lines = [next(f) for x in range(nrows)]

    df = pd.DataFrame(lines)
    df.columns = ["name"]
    df[feat] = df.name.str.split("\t", expand=True,)
    df.drop(columns=["name"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_data(
    train_file="../public_dat/train.data",
    validation_file="../public_dat/validation.data",
    test_file="../public_dat/test.data",
    feat_name_file="../public_dat/feature.name",
    sol_file="../public_dat/df.solution",
    nrows=None,
):
    logging.info("loading feature names...")
    feature_names = open(feat_name_file).readlines()[0].split("\t")
    logging.info("loading labels...")
    labels = list(map(lambda x: float(x), open(sol_file).readlines()))
    logging.info("loading training data...")
    df = read_csv(train_file, feature_names, nrows)
    df.drop(columns="tweet_id", inplace=True)
    if nrows is not None:
        labels = labels[: df.shape[0]]

    logging.info("loading validation data...")
    validation = read_csv(validation_file, feature_names, nrows)
    validation.drop(columns="tweet_id", inplace=True)

    logging.info("loading test data...")
    feature_names.remove("tweet_id")
    test = read_csv(test_file, feature_names, nrows)

    return df, labels, validation, test


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

    return " ".join(tokens)


def gen_feature(df):
    df["entities_isnull"] = df["entities"].map(
        lambda x: 1 if x.startswith("null;") else 0
    )
    df["mentions_isnull"] = df["mentions"].map(
        lambda x: 1 if x.startswith("null;") else 0
    )
    df["hashtags_isnull"] = df["hashtags"].map(
        lambda x: 1 if x.startswith("null;") else 0
    )
    df["urls_isnull"] = df["urls"].map(lambda x: 1 if x.startswith("null;") else 0)

    df["parsed_entity"] = df["entities"].map(
        lambda x: ""
        if x.startswith("null;")
        else " ".join([y.split(":")[0] for y in x.split(";")[:-1]])
    )
    df["entity_char_len"] = df["parsed_entity"].map(lambda x: len(x))

    df["min_ts"] = df.groupby("username")["timestamp"].transform("min")
    df["max_ts"] = df.groupby("username")["timestamp"].transform("max")
    df["duration"] = (df["max_ts"] - df["min_ts"]).map(lambda x: x.days)

    df["#favorites"] = df["#favorites"].astype(np.int64)
    df["mentions_favorite_mean"] = df.groupby("mentions")["#favorites"].transform(
        "mean"
    )

    df["user_mention_nunique"] = df.groupby("username")["mentions"].transform("nunique")
    df["user_hashtag_nunique"] = df.groupby("username")["hashtags"].transform("nunique")
    df["user_url_nunique"] = df.groupby("username")["urls"].transform("nunique")
    df["user_entity_nunique"] = df.groupby("username")["entities"].transform("nunique")

    return df


def main(
    train_file,
    valid_file,
    test_file,
    feat_name_file,
    sol_file,
    train_feature_file,
    valid_feature_file,
    test_feature_file,
):
    nrows = None
    train, labels, validation, test = load_data(
        train_file, valid_file, test_file, feat_name_file, sol_file, nrows=nrows,
    )
    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2
    train["split"] = TRN_SET
    validation["split"] = VAL_SET
    test["split"] = TST_SET

    train["LABEL"] = labels
    validation["LABEL"] = np.nan
    test["LABEL"] = np.nan

    logging.info("engineering features")
    df = pd.concat((train, validation, test))
    df = gen_feature(df)

    features = [
        "entities_isnull",
        "mentions_isnull",
        "hashtags_isnull",
        "urls_isnull",
        "duration",
        "entity_char_len",
        "mentions_favorite_mean",
        "user_mention_nunique",
        "user_hashtag_nunique",
        "user_url_nunique",
        "user_entity_nunique",
    ]
    train = df[df["split"] == TRN_SET]
    validation = df[df["split"] == VAL_SET]
    test = df[df["split"] == TST_SET]

    logging.info("saving training features")
    train[features].to_csv(train_feature_file, index=False, compression="gzip")
    logging.info("saving validation features")
    validation[features].to_csv(valid_feature_file, index=False, compression="gzip")
    logging.info("saving testing featured")
    test[features].to_csv(test_feature_file, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, dest="train_file")
    parser.add_argument("--valid-file", required=True, dest="valid_file")
    parser.add_argument("--test-file", required=True, dest="test_file")
    parser.add_argument("--feat-name-file", required=True, dest="feat_name_file")
    parser.add_argument("--sol-file", required=True, dest="sol_file")
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
        args.train_file,
        args.valid_file,
        args.test_file,
        args.feat_name_file,
        args.sol_file,
        args.train_feature_file,
        args.valid_feature_file,
        args.test_feature_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

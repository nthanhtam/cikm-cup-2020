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
    t = pd.crosstab(index=df["user_enc"], columns=df["negative"])
    t.columns = ["neg_{}".format(i) for i in t.columns]
    t["std_neg"] = t.std(axis=1)
    df = df.merge(t, how="left", on="user_enc")

    t = pd.crosstab(index=df["user_enc"], columns=df["positive"])
    t.columns = ["pos_{}".format(i) for i in t.columns]
    t["std_pos"] = t.std(axis=1)
    df = df.merge(t, how="left", on="user_enc")

    return df


def main(
    train_file,
    valid_file,
    test_file,
    train_feature_file,
    valid_feature_file,
    test_feature_file,
):
    nrows = None

    logging.info("loading train features")
    train = pd.read_csv(train_file, usecols=["user_enc", "positive", "negative"])
    logging.info("loading valid features")
    validation = pd.read_csv(valid_file, usecols=["user_enc", "positive", "negative"])
    logging.info("loading test features")
    test = pd.read_csv(test_file, usecols=["user_enc", "positive", "negative"])

    TRN_SET = 0
    VAL_SET = 1
    TST_SET = 2
    train["split"] = TRN_SET
    validation["split"] = VAL_SET
    test["split"] = TST_SET

    logging.info("engineering features")
    df = pd.concat((train, validation, test))
    df = gen_feature(df)

    train = df[df["split"] == TRN_SET].drop(
        columns=["user_enc", "positive", "negative"]
    )
    validation = df[df["split"] == VAL_SET].drop(
        columns=["user_enc", "positive", "negative"]
    )
    test = df[df["split"] == TST_SET].drop(columns=["user_enc", "positive", "negative"])

    logging.info("saving training features")
    train.to_csv(train_feature_file, index=False, compression="gzip")
    logging.info("saving validation features")
    validation.to_csv(valid_feature_file, index=False, compression="gzip")
    logging.info("saving testing featured")
    test.to_csv(test_feature_file, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, dest="train_file")
    parser.add_argument("--valid-file", required=True, dest="valid_file")
    parser.add_argument("--test-file", required=True, dest="test_file")
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
        args.train_feature_file,
        args.valid_feature_file,
        args.test_feature_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

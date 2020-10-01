#!/usr/bin/env python

__author__ = "Mastercard Team"
__email__ = ""
__contributors__ = ["Mastercard Team", "Tam Nguyen"]


import argparse
import logging
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

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


def gen_feature(df):
    df["hashtags"] = df["hashtags"].map(lambda x: "" if x == "null;" else x)
    df["hashtags"] = df["hashtags"].fillna("")
    df["Date"] = df["timestamp"].map(lambda x: x.date())
    grouped_hashtags = (
        df.groupby("Date")["hashtags"].apply(lambda x: " ".join(x)).reset_index()
    )

    vectorizer = CountVectorizer(binary=True)
    hashtag_age = vectorizer.fit_transform(grouped_hashtags["hashtags"]).todense()

    hastag_life = csr_matrix(hashtag_age.sum(axis=0))
    for i in range(1, len(hashtag_age)):
        temp_old = hashtag_age[i - 1]
        temp = hashtag_age[i]
        new_temp = np.where(temp_old > 0, temp_old + 1, 0)
        new_temp = np.where((temp_old == 0) & (temp == 1), 1, new_temp)
        hashtag_age[i] = new_temp

    weight_hashtag = np.where(hashtag_age > 0, 1 / np.log1p(hashtag_age), 0)

    hashtag_age = csr_matrix(hashtag_age)
    weight_hashtag = csr_matrix(weight_hashtag)

    Y = vectorizer.transform(df["hashtags"])
    Y_bool = Y.astype(bool)
    df["avg_score"] = 0
    df["max_score"] = 0
    df["count_for_score"] = 0
    df["avg_weighted_score"] = 0
    df["max_weighted_score"] = 0
    df["avg_hashtag_age"] = 0
    df["max_hashtag_age"] = 0
    df["min_hashtag_age"] = 0
    df["max_hashtag_life"] = 0
    df["avg_hashtag_life"] = 0

    ht_count = CountVectorizer().fit_transform(grouped_hashtags["hashtags"])

    for date in tqdm(df["Date"].unique()):
        row_index = df["Date"] == date
        val_index = grouped_hashtags["Date"] == date
        Z = Y_bool[row_index.values].multiply(ht_count[val_index.values])
        Z_weighted = Z.multiply(weight_hashtag[val_index.values])
        weights = Y_bool[row_index.values].multiply(weight_hashtag[val_index.values])
        Z_hashtag = Y_bool[row_index.values].multiply(hashtag_age[val_index.values])
        Z_hastag_life = Y_bool[row_index.values].multiply(hastag_life)
        sums = Z.sum(axis=1).A1
        sum_weighted = Z_weighted.sum(axis=1).A1
        weight = weights.sum(axis=1).A1
        counts = np.diff(Z.indptr)
        averages = sums / counts
        df["avg_score"].iloc[row_index.values] = averages
        df["max_score"].iloc[row_index.values] = Z.max(axis=1).A.ravel()
        df["avg_weighted_score"].iloc[row_index.values] = sum_weighted / weight
        df["max_weighted_score"].iloc[row_index.values] = Z_weighted.max(
            axis=1
        ).A.ravel()
        df["avg_hashtag_age"].iloc[row_index.values] = Z_hashtag.sum(
            axis=1
        ).A1 / np.diff(Z_hashtag.indptr)
        df["max_hashtag_age"].iloc[row_index.values] = Z_hashtag.max(axis=1).A.ravel()
        df["min_hashtag_age"].iloc[row_index.values] = weights.max(axis=1).A.ravel()
        df["max_hashtag_life"].iloc[row_index.values] = Z_hastag_life.max(
            axis=1
        ).A.ravel()
        df["avg_hashtag_life"].iloc[row_index.values] = Z_hastag_life.sum(
            axis=1
        ).A1 / np.diff(Z_hastag_life.indptr)
        df["count_for_score"].iloc[row_index.values] = counts

    df["avg_score"] = np.where(df["avg_score"].isnull(), 0, df["avg_score"])
    df["max_score"] = np.where(df["max_score"].isnull(), 0, df["max_score"])
    df["count_for_score"] = np.where(
        df["count_for_score"].isnull(), 0, df["count_for_score"]
    )
    df["avg_weighted_score"] = np.where(
        df["avg_weighted_score"].isnull(), 0, df["avg_weighted_score"]
    )
    df["max_weighted_score"] = np.where(
        df["max_weighted_score"].isnull(), 0, df["max_weighted_score"]
    )
    df["avg_hashtag_age"] = np.where(
        df["avg_hashtag_age"].isnull(), 0, df["avg_hashtag_age"]
    )
    df["max_hashtag_age"] = np.where(
        df["max_hashtag_age"].isnull(), 0, df["max_hashtag_age"]
    )
    df["min_hashtag_age"] = np.where(
        df["min_hashtag_age"].isnull(), 0, df["min_hashtag_age"]
    )
    df["max_hashtag_life"] = np.where(
        df["max_hashtag_life"].isnull(), 0, df["max_hashtag_life"]
    )
    df["avg_hashtag_life"] = np.where(
        df["avg_hashtag_life"].isnull(), 0, df["avg_hashtag_life"]
    )

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
        "avg_weighted_score",
        "max_weighted_score",
        "avg_hashtag_age",
        "max_hashtag_age",
        "min_hashtag_age",
        "max_hashtag_life",
        "avg_hashtag_life",
        "max_score",
        "count_for_score",
        "avg_score",
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

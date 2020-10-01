#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "nthanhtam@gmail.com"

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


def preprocess(df):
    df[["positive", "negative"]] = (
        df["sentiment"].str.split(" ", expand=True).applymap(int)
    )
    df["hashtags"] = df["hashtags"].map(
        lambda x: "covid19"
        if x
        in [
            "covid",
            "covid19,",
            "covid2019",
            "covidー19",
            "coronavirus",
            "covid-19",
            "covid19.",
            "covid_19",
            "coronavirus covid19",
            "coronavirus.",
            "covid19 coronavirus",
        ]
        else x
    )

    df["tweet_id"] = range(df.shape[0])

    return df


def gen_text_feature(df):
    df["parsed_entities"] = df["entities"].map(lambda x: parse_entity(x))
    X_e = CountVectorizer().fit_transform(df["parsed_entities"])
    svd = TruncatedSVD(n_components=5)
    X_svd = svd.fit_transform(X_e)

    for i in range(X_svd.shape[1]):
        df[f"entity_svd{i}"] = X_svd[:, i]

    X_m = CountVectorizer().fit_transform(df["mentions"])
    svd = TruncatedSVD(n_components=5)
    X_svd = svd.fit_transform(X_m)

    for i in range(X_svd.shape[1]):
        df[f"mentions_svd{i}"] = X_svd[:, i]

    X_h = CountVectorizer().fit_transform(df["hashtags"])

    svd = TruncatedSVD(n_components=5)
    X_svd = svd.fit_transform(X_h)

    for i in range(X_svd.shape[1]):
        df[f"hashtag_svd{i}"] = X_svd[:, i]

    X_u = CountVectorizer().fit_transform(df["urls"])
    svd = TruncatedSVD(n_components=5)
    X_svd = svd.fit_transform(X_u)

    for i in range(X_svd.shape[1]):
        df[f"url_svd{i}"] = X_svd[:, i]

    return df


def gen_feature(df):
    df["hashtags_enc"] = LabelEncoder().fit_transform(df["hashtags"].map(str))
    df["mentions_enc"] = LabelEncoder().fit_transform(df["mentions"].map(str))
    df["user_enc"] = LabelEncoder().fit_transform(df["username"].map(str))
    user_cnt = df["username"].value_counts().to_frame(name="user_cnt")
    user_cnt = user_cnt.reset_index()
    user_cnt.rename(columns={"index": "username"}, inplace=True)
    df = df.merge(user_cnt, how="left", on="username")
    df["url_len"] = df["urls"].map(lambda x: len(str(x)))
    df["entities_len"] = df["entities"].map(
        lambda x: 0 if x == "null;" else len(str(x).split(";"))
    )

    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    df["date"] = df["timestamp"].dt.date

    dt_cnt = (
        df.groupby("date")["tweet_id"].count().to_frame(name="dt_cnt").reset_index()
    )
    df = df.merge(dt_cnt, how="left", on="date")

    df[["#followers", "#favorites", "#friends"]] = df[
        ["#followers", "#favorites", "#friends"]
    ].astype(int)

    df["followers/friends"] = df["#followers"] / df["#friends"]
    df["favorites/followers"] = df["#favorites"] / df["#followers"]
    df["favorites/friends"] = df["#favorites"] / df["#friends"]

    user_stats = df.groupby("username")[["#followers", "#friends", "#favorites"]].agg(
        [np.mean, np.median, np.sum, np.min, np.max, np.std]
    )
    user_stats.columns = [f"{c1}_{c2}" for c1, c2 in user_stats.columns]
    user_stats.reset_index(inplace=True)

    df = df.merge(user_stats, how="left", on="username")

    dt_cnt = df.groupby("date")["username"].count()
    dt_cnt = dt_cnt.to_frame(name="daily_cnt").reset_index()
    df = df.merge(dt_cnt, how="left", on="date")

    hr_cnt = df.groupby(["date", "hour"])["username"].count()
    hr_cnt = hr_cnt.to_frame(name="hourly_cnt").reset_index()

    df = df.merge(hr_cnt, how="left", on=["date", "hour"])

    hr_cnt = df.groupby(["username", "date", "hour"])["username"].count()
    hr_cnt = hr_cnt.to_frame(name="user_hourly_cnt").reset_index()
    df = df.merge(hr_cnt, how="left", on=["username", "date", "hour"])

    hr_cnt = df.groupby(["username", "date"])["username"].count()
    hr_cnt = hr_cnt.to_frame(name="user_daily_cnt").reset_index()
    df = df.merge(hr_cnt, how="left", on=["username", "date"])

    df["hashtags_cnt"] = df["hashtags"].map(
        lambda x: 0 if x == "null;" else len(x.split(" "))
    )

    hr_cnt = df.groupby(["username", "month"])["username"].count()
    hr_cnt = hr_cnt.to_frame(name="user_monthly_cnt").reset_index()
    df = df.merge(hr_cnt, how="left", on=["username", "month"])

    df["mention_cnt"] = df["mentions"].apply(lambda x: len(str(x).split(" ")))
    df["mention_cnt"] = np.where(df["mentions"] == "null;", 0, df["mention_cnt"])

    cnt = df.groupby("hashtags")["tweet_id"].count()
    cnt = cnt.to_frame(name="hashtags_tweet_cnt").reset_index()
    df = df.merge(cnt, how="left", on="hashtags")

    cnt = df.groupby("mentions")["tweet_id"].count()
    cnt = cnt.to_frame(name="mentions_tweet_cnt").reset_index()
    df = df.merge(cnt, how="left", on="mentions")

    cnt = df.groupby("urls")["tweet_id"].count()
    cnt = cnt.to_frame(name="urls_tweet_cnt").reset_index()
    df = df.merge(cnt, how="left", on="urls")

    user_stats = df.groupby(["username", "date"])[["#favorites"]].agg(
        [np.mean, np.sum, np.min, np.max, np.std]
    )
    user_stats.columns = [
        f"user_daily_%s_%s" % (c1, c2) for c1, c2 in user_stats.columns
    ]
    df = df.merge(user_stats, how="left", on=["username", "date"])

    nuique_hour = df.groupby("username")["hour"].nunique()
    nuique_hour = nuique_hour.to_frame(name="hour_set_len").reset_index()
    df = df.merge(nuique_hour, how="left", on="username")

    nuique_hour = df.groupby("username")["weekday"].nunique()
    nuique_hour = nuique_hour.to_frame(name="weekday_set_len").reset_index()
    df = df.merge(nuique_hour, how="left", on="username")

    ht_stats = (
        df.loc[df["hashtags"] != "null;"]
        .groupby("hashtags")[["#favorites"]]
        .agg([np.mean, np.sum, np.min, np.max, np.std])
    )
    ht_stats.columns = [f"hashtags_{c1}_{c2}" for c1, c2 in ht_stats.columns]
    df = df.merge(ht_stats, how="left", on=["hashtags"])

    user_stats = df.groupby(["month"])[["#favorites"]].agg(
        [np.mean, np.sum, np.min, np.max, np.std]
    )
    user_stats.columns = [
        f"monthly_{c1}_{c2}" % (c1, c2) for c1, c2 in user_stats.columns
    ]
    df = df.merge(user_stats, how="left", on=["month"])

    df["user_positive_mean"] = df.groupby("username")["positive"].transform("mean")
    df["user_negative_mean"] = df.groupby("username")["negative"].transform("mean")

    df["user_positive_std"] = df.groupby("username")["positive"].transform("std")
    df["user_negative_std"] = df.groupby("username")["negative"].transform("std")

    df["followers_positive"] = df["#followers"] * df["positive"]
    df["friends_positive"] = df["#friends"] * df["positive"]

    df["followers_negative"] = df["#followers"] * np.abs(df["negative"])
    df["friends_negative"] = df["#friends"] * np.abs(df["negative"])

    df["favor_cnt"] = df.groupby("#favorites")["tweet_id"].transform("count")
    df["follo_cnt"] = df.groupby("#followers")["tweet_id"].transform("count")
    df["frien_cnt"] = df.groupby("#friends")["tweet_id"].transform("count")

    df["user_hashtag_cnt"] = df.groupby(["username", "hashtags"])["tweet_id"].transform(
        "count"
    )
    df["user_mention_cnt"] = df.groupby(["username", "mentions"])["tweet_id"].transform(
        "count"
    )
    df["user_hashtag_mention_cnt"] = df.groupby(["username", "hashtags", "mentions"])[
        "tweet_id"
    ].transform("count")

    df["user_entity_cnt"] = df.groupby(["username", "entities"])["tweet_id"].transform(
        "count"
    )
    df["user_urls_cnt"] = df.groupby(["username", "urls"])["tweet_id"].transform(
        "count"
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

    logging.info("preprocessing features")
    df = pd.concat((train, validation, test))
    df = preprocess(df)
    logging.info("engineering features")
    df = gen_feature(df)
    logging.info("generating text features")
    df = gen_text_feature(df)

    sel_features = [
        "#followers",
        "#friends",
        "#favorites",
        "positive",
        "negative",
        "hour",
        "weekday",
        "month",
        "user_enc",
        "user_cnt",
        "hashtags_enc",
        "mentions_enc",
        "url_len",
        "entities_len",
        "dt_cnt",
        "#followers_mean",
        "#followers_median",
        "#followers_sum",
        "#followers_amin",
        "#followers_amax",
        "#followers_std",
        "#friends_mean",
        "#friends_median",
        "#friends_sum",
        "#friends_amin",
        "#friends_amax",
        "#friends_std",
        "#favorites_mean",
        "#favorites_median",
        "#favorites_sum",
        "#favorites_amin",
        "#favorites_amax",
        "#favorites_std",
        "daily_cnt",
        "hourly_cnt",
        "hashtags_cnt",
        "user_hourly_cnt",
        "favorites/followers",
        "favorites/friends",
        "user_monthly_cnt",
        "mention_cnt",
        "hashtags_tweet_cnt",
        "mentions_tweet_cnt",
        "urls_tweet_cnt",
        "hour_set_len",
        "weekday_set_len",
        "user_positive_mean",
        "user_negative_mean",
        "user_positive_std",
        "user_negative_std",
        "followers_positive",
        "friends_positive",
        "followers_negative",
        "friends_negative",
        "entity_svd0",
        "entity_svd1",
        "entity_svd2",
        "entity_svd3",
        "entity_svd4",
        "mentions_svd0",
        "mentions_svd1",
        "mentions_svd2",
        "mentions_svd3",
        "mentions_svd4",
        "hashtag_svd0",
        "hashtag_svd1",
        "hashtag_svd2",
        "hashtag_svd3",
        "hashtag_svd4",
        "url_svd0",
        "url_svd1",
        "url_svd2",
        "url_svd3",
        "url_svd4",
    ]

    train = df[df["split"] == TRN_SET]
    validation = df[df["split"] == VAL_SET]
    test = df[df["split"] == TST_SET]

    logging.info("saving training features")
    train[["LABEL"] + sel_features].to_csv(
        train_feature_file, index=False, compression="gzip"
    )
    logging.info("saving validation features")
    validation[sel_features].to_csv(valid_feature_file, index=False, compression="gzip")
    logging.info("saving testing featured")
    test[sel_features].to_csv(test_feature_file, index=False, compression="gzip")


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

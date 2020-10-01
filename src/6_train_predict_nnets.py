#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "tam@aidatech.io"

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import pickle
import lightgbm as lgb
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    Input,
    Embedding,
    Dropout,
    concatenate,
    Flatten,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def make_X(df):
    X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
        X[v] = df[[v]].to_numpy()
    return X


def create_model(
    lr=0.002,
    vocab_size=500000 + 1,
    max_length=15,
    max_length_netloc=5,
    max_length_path=50,
    seed=123,
):
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    # Dense input
    dense_input = Input(shape=(len(dense_cols),), name="dense1")
    w2v_input = Input(shape=(len(emb_cols),), name="w2v")

    # Embedding input
    hour_input = Input(shape=(1,), name="hour")
    month_input = Input(shape=(1,), name="month")
    weekday_input = Input(shape=(1,), name="weekday")
    user_input = Input(shape=(1,), name="user_enc")
    hashtag_input = Input(shape=(1,), name="hashtags_enc")
    mention_input = Input(shape=(1,), name="mentions_enc")
    entity_input = Input(shape=(max_length,), name="entities")
    netloc_input = Input(shape=(max_length_netloc,), name="netloc")
    path_input = Input(shape=(max_length_path,), name="path")

    hour_emb = Flatten()(Embedding(24, 1)(hour_input))
    month_emb = Flatten()(Embedding(13, 1)(month_input))
    weekday_emb = Flatten()(Embedding(7, 1)(weekday_input))
    user_emb = Flatten()(Embedding(4298559, 1)(user_input))
    hashtag_emb = Flatten()(Embedding(1573918, 1)(hashtag_input))
    mention_emb = Flatten()(Embedding(2394565, 1)(mention_input))
    entity_emb = Flatten()(Embedding(vocab_size, 25)(entity_input))
    netloc_emb = Flatten()(Embedding(vocab_size, 15)(netloc_input))
    path_emb = Flatten()(Embedding(vocab_size, 25)(path_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate(
        [
            dense_input,
            w2v_input,
            hour_emb,
            month_emb,
            weekday_emb,
            user_emb,
            hashtag_emb,
            mention_emb,
            entity_emb,
            netloc_emb,
            path_emb,
        ]
    )

    x = Dense(
        1024,
        activation="tanh",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
    )(x)

    x = Dropout(0.5)(x)
    x = Dense(
        512,
        activation="tanh",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
    )(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="tanh")(x)
    outputs = Dense(1, activation="linear", name="output")(x)

    inputs = {
        "dense": dense_input,
        "w2v": w2v_input,
        "hour": hour_input,
        "month": month_input,
        "weekday": weekday_input,
        "user_enc": user_input,
        "hashtags_enc": hashtag_input,
        "mentions_enc": mention_input,
        "entities": entity_input,
        "netloc": netloc_input,
        "path": path_input,
    }

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(
        loss=keras.losses.mean_squared_error,
        metrics=["mse"],
        optimizer=keras.optimizers.RMSprop(learning_rate=lr),
    )
    return model


num_cols = [
    "#followers",
    "#friends",
    "#favorites",
    "positive",
    "negative",
    "user_cnt",
    "url_len",
    "entities_len",
    "dt_cnt",
    "daily_cnt",
    "hourly_cnt",
    "hashtags_cnt",
    "user_hourly_cnt",
    "user_monthly_cnt",
    "mention_cnt",
    "hashtags_tweet_cnt",
    "mentions_tweet_cnt",
    "urls_tweet_cnt",
    "hour_set_len",
    "weekday_set_len",
    "followers_positive",
    "friends_positive",
    "followers_negative",
    "friends_negative",
]
txt_cols = [
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

stats_cols = [
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
    "user_positive_mean",
    "user_negative_mean",
    "user_positive_std",
    "user_negative_std",
    "followers_positive",
    "friends_positive",
    "followers_negative",
    "friends_negative",
]
cat_cols = [
    "hour",
    "weekday",
    "month",
    "user_enc",
    "hashtags_enc",
    "mentions_enc",
]


def preprocess(X, X_tst):
    for c in num_cols:
        if c == "negative":
            X[c] = np.log1p(-X[c])
            X_tst[c] = np.log1p(-X_tst[c])
        else:
            X[c] = np.log1p(X[c])
            X_tst[c] = np.log1p(X_tst[c])
    return X, X_tst


def main(
    train_feature_file,
    valid_feature_file,
    test_feature_file,
    train_predict_file,
    valid_predict_file,
    test_predict_file,
    valid_id_file,
):
    logging.info("loading train features")
    X = pd.read_csv(train_feature_file)
    y = X["LABEL"]
    X.drop(columns="LABEL", inplace=True)

    logging.info("loding valid features")
    X_tst = pd.read_csv(valid_feature_file)

    model = create_model(0.0002)
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    history = model.fit(
        X_train,
        y_train,
        batch_size=10000,
        epochs=30,
        shuffle=True,
        validation_data=(X_valid, y_valid),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
    )

    val_pred = predict(model, X_val)
    tst_pred = predict(model, X_tst)

    np.savetxt(valid_predict_file, val_pred, fmt="%0.5")
    np.savetxt(test_predict_file, tst_pred, fmt="%0.5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-feature-file", required=True, dest="train_feature_file"
    )
    parser.add_argument(
        "--valid-feature-file", required=True, dest="valid_feature_file"
    )
    parser.add_argument("--test-feature-file", required=True, dest="test_feature_file")

    parser.add_argument(
        "--train-predict-file", required=True, dest="train_predict_file"
    )
    parser.add_argument(
        "--valid-predict-file", required=True, dest="valid_predict_file"
    )
    parser.add_argument("--test-predict-file", required=True, dest="test_predict_file")
    parser.add_argument("--valid-id-file", required=True, dest="valid_id_file")

    args = parser.parse_args()
    start = time.time()
    main(
        args.train_feature_file,
        args.valid_feature_file,
        args.test_feature_file,
        args.train_predict_file,
        args.valid_predict_file,
        args.test_predict_file,
        args.valid_id_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

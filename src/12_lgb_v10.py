#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "nthanhtam@gmail.com"

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle
import time
from scipy.sparse import hstack, csc_matrix
import lightgbm as lgb

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True, dest="feature_file")
    parser.add_argument("--valid-pred-file", required=True, dest="valid_pred_file")
    parser.add_argument("--test-pred-file", required=True, dest="test_pred_file")
    parser.add_argument("--random-state", required=False, type=int, dest="random_state")

    args = parser.parse_args()
    random_state = args.random_state
    logging.info(f"random state: {random_state}")

    start = time.time()

    logging.info("loading data...")
    X, y, X_tst = pickle.load(open(args.feature_file, "rb"))

    X_v5 = pd.read_csv("./features/train_v5.csv.gz")
    X_tst_v5 = pd.read_csv("./features/valid_v5.csv.gz")

    X = pd.concat((X, X_v5), axis=1)
    X_tst = pd.concat((X_tst, X_tst_v5), axis=1)

    params = {
        "objective": "regression",
        "num_leaves": 35,
        "max_depth": 10,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "learning_rate": 0.2,
        "metric": "rmse",
    }
    model = lgb.LGBMRegressor(**params)
    model.set_params(
        **{"metric": "mse", "n_estimators": 15000, "random_state": args.random_state}
    )

    logging.info("training models...")
    tst_pred = 0
    for k, rt in enumerate([7, 111, 777, 773]):
        t = X.copy()
        t["LABEL"] = y
        pos = t[t["LABEL"] > 0]
        neg = t[t["LABEL"] == 0]
        neg = neg.sample(frac=0.90, random_state=123 + rt)
        t = pd.concat((pos, neg))
        X_trn1 = t.drop(columns="LABEL")
        y_trn1 = t["LABEL"]
        print(X.shape, X_trn1.shape)

        model.fit(
            X_trn1, np.log1p(y_trn1), eval_set=[(X_trn1, np.log1p(y_trn1))], verbose=500
        )
        tst_pred += np.expm1(model.predict(X_tst))

        np.savetxt(args.test_pred_file, tst_pred / (k + 1), fmt="%0.5f")

    tst_pred = tst_pred / (k + 1)

    np.savetxt(args.test_pred_file, tst_pred, fmt="%0.5f")

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

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
from scipy.sparse import hstack, csc_matrix

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


model = lgb.LGBMRegressor(**params)
model.set_params(**{"metric": "rmse", "n_estimators": 10000})

model.fit(
    X_trn,
    np.log1p(y_trn),
    eval_set=[(X_trn, np.log1p(y_trn)), (X_val, np.log1p(y_val))],
    early_stopping_rounds=10,
    verbose=50,
    eval_metric="rmse",
)



X_tst = hstack((X_tst, cnt_feat_tst), format="csr")


tst_pred = model.predict(X_tst)

tst_pred = np.expm1(tst_pred)
tst_pred = np.where(tst_pred < 1, 0, tst_pred)


np.savetxt("../sub/validation.predict", np.round(tst_pred), fmt="%i")


with open("../public_dat/feat_v2.pkl", "wb") as f:
    pickle.dump([X, y, X_tst], f, protocol=4)


with open("../public_dat/feat.pkl", "rb") as f:
    X, y, X_tst = pickle.load(f)


model.set_params(**{"n_estimtors": 4000})



model.fit(X, np.log1p(y), verbose=200)

tst_pred = model.predict(X_tst)
tst_pred = np.expm1(tst_pred)
tst_pred = np.where(tst_pred < 1, 0, tst_pred)


np.savetxt("../sub/validation.predict", np.round(tst_pred), fmt="%i")


def load_data(
    train_feature_file, valid_feature_file, test_feature_file,
):
    train = pd.read_csv(train_feature_file)
    valid = pd.read_csv(valid_feature_file)
    test = pd.read_csv(test_feature_file)

    return train, valid, test


def predict(model, X):
    pred = model.predict(X)
    pred = np.expm1(pred)
    return pred


def train_model(model, X, y, cv_id):
    n_folds = np.max(cv_id)
    if n_folds == 1:
        X_trn = X.iloc[cv_id == 1]
        X_val = X.iloc[cv_id != 1]
        y_trn = y[cv_id == 1]
        y_val = y[cv_id != 1]

        model.fit(
            X_trn,
            np.log1p(y_trn),
            eval_set=[(X_trn, np.log1p(y_trn)), (X_val, np.log1p(y_val))],
            early_stopping_rounds=10,
            verbose=50,
            eval_metric="rmse",
        )
        return model
    else:
        pass


def main(
    train_feature_file,
    valid_feature_file,
    test_feature_file,
    train_predict_file,
    valid_predict_file,
    test_predict_file,
    valid_id_file,
):
    X, X_val, X_tst = load_data(
        train_feature_file, valid_feature_file, test_feature_file
    )

    cv_id = np.loadtxt(valid_id_file)
    y = X["LABEL"]
    X.drop(columns="LABEL", inplace=True)

    params = {
        "objective": "regression",
        "num_leaves": 35,
        "max_depth": 10,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "learning_rate": 0.2,
    }
    model = lgb.LGBMRegressor(**params)
    model.set_params(**{"metric": "rmse", "n_estimators": 4000})

    model = train_model(model, X, y, cv_id)

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

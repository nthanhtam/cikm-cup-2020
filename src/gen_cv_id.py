import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import logging

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
    train_file="public_dat/train.data",
    feat_name_file="public_dat/feature.name",
    nrows=None,
):
    logging.info("loading feature names...")
    feature_names = open(feat_name_file).readlines()[0].split("\t")
    logging.info("loading training data...")
    df = read_csv(train_file, feature_names, nrows)
    df = df[["timestamp"]]
    return df


if __name__ == "__main__":
    df = load_data()
    # 2019-09-30 22:00:00+00:00 2020-04-30 21:59:46+00:00
    df["cv_id"] = 1
    df.loc[df["timestamp"] >= pd.to_datetime("2020-03-31 23:59:00+00:00"), "cv_id"] = 0

    np.savetxt("date_holdout_cv_id.txt", df["cv_id"].values, fmt="%i")

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_id = np.zeros(df.shape[0])
    for k, (trn_id, val_id) in enumerate(kf.split(df)):
        cv_id[val_id] = k

    np.savetxt("kfold_cv_id.txt", cv_id, fmt="%i")

    indices = np.arange(df.shape[0])
    x1, x2, idx1, idx2 = train_test_split(df, indices, test_size=0.2, random_state=123)
    cv_id = np.zeros(df.shape[0])
    cv_id[idx1] = 1
    np.savetxt("random_holdout_cv_id.txt", cv_id, fmt="%i")

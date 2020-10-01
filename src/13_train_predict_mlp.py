import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict
import logging
import tensorflow.keras as ks
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

def read_csv(file_name, feat, nrows=None):
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
    train_file="./public_dat/train.data",
    validation_file="./public_dat/validation.data",
    test_file="./public_dat/test.data",
    feat_name_file="./public_dat/feature.name",
    sol_file="./public_dat/train.solution",
    nrows=None,
):
    logging.info("loading feature names...")
    feature_names = open(feat_name_file).readlines()[0].split("\t")
    logging.info("loading labels...")
    labels = list(map(lambda x: float(x), open(sol_file).readlines()))
    logging.info("loading training data...")
    
    df = pd.read_csv(train_file, header=None, sep='\t', 
                     quoting=csv.QUOTE_NONE, 
                     names=feature_names, 
                     #parse_dates=['timestamp'], 
                     nrows=nrows)

    df.drop(columns="tweet_id", inplace=True)
    if nrows is not None:
        labels = labels[: df.shape[0]]

    logging.info("loading validation data...")
    validation = pd.read_csv(validation_file, header=None, sep='\t', 
                             quoting=csv.QUOTE_NONE, 
                             names=feature_names, 
                             #parse_dates=['timestamp'], 
                             nrows=nrows)

    validation.drop(columns="tweet_id", inplace=True)

    logging.info("loading test data...")
    feature_names.remove("tweet_id")
    #test = read_csv(test_file, feature_names, nrows)
    test = pd.read_csv(test_file, header=None, sep='\t', 
                       quoting=csv.QUOTE_NONE, 
                       names=feature_names, 
                       #parse_dates=['timestamp'], 
                       nrows=nrows)

    return df, labels, validation, test

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['entities'].fillna('')
    df['text'] = (df['hashtags'].fillna('') + ' ' + df['mentions'].fillna('') + ' ' + df['urls'].fillna(''))
    df['#fiends'] = np.log1p(df['#friends'])
    df['#followers'] = np.log1p(df['#followers'])
    df['#favorites'] = np.log1p(df['#favorites'])
    return df[['name', 'text', '#friends', '#followers','#favorites']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    batch_size = 2048
    X_train, X_test = xs
    tf.random.set_seed(173)
    tf.keras.backend.clear_session()

    model_in = ks.layers.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True, batch_size=batch_size)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    for i in range(3):
        with timer(f'epoch {i + 1}'):
            model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1, verbose=0)

    return model.predict(X_test)[:, 0]

def main():
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=1000)),
        on_field('text', Tfidf(max_features=1000)),
        on_field(['#friends', '#followers','#favorites'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()
    with timer('process train'):
        train, label, __, __ = load_data()
        train['label'] = label
        #train = train.sample(10000)
        
        train, valid = train_test_split(train, test_size=0.2, random_state=123)

        y_train = y_scaler.fit_transform(np.log1p(train['label'].values.reshape(-1, 1)))
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')

        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    with timer('fit predict'):
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] 
        #y_pred0 = fit_predict(xs[0], y_train=y_train)
        y_pred1 = fit_predict(xs[1], y_train=y_train)
        y_pred = y_pred1
        
        y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
        print('Valid MSLE: {:.4f}'.format(mean_squared_log_error(valid['label'], np.where(y_pred<0,0,y_pred))))

if __name__ == '__main__':
    main()
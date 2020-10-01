#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "nthanhtam@gmail.com"

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import logging
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import time

logging.basicConfig(
    format="%(asctime)s   %(levelname)s   %(message)s", level=logging.DEBUG
)


def create_user_corpus(df, col="hashtags"):
    t = df[df[col] != ""][["username", col]]
    t = t.drop_duplicates(subset=["username", col])
    t = t.groupby("username")[col].agg({col: " ".join})
    t = t[t[col].str.strip() != ""]
    t.to_csv(f"./features/user_{col}_corpus.csv")


def load_corpus(col="hashtags"):
    filename = f"./features/user_{col}_corpus.csv"
    t = pd.read_csv(filename)
    print(f"corpus size: {t.shape[0]}")
    docs = []
    for i, row in tqdm(t.iterrows()):
        uids = row["username"]
        sent = row[col]
        doc = TaggedDocument(sent.split(" "), [uids])
        docs.append(doc)

    return docs


def train_doc2vec(train_corpus, col="hashtags", n_workers=10):
    model = Doc2Vec(
        vector_size=64, min_count=10, epochs=5, window=5, workers=n_workers,
    )
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def extract_feature(model, col="hashtags"):
    filename = f"./features/user_{col}_corpus.csv"
    t = pd.read_csv(filename)

    ec = 0
    res = {}
    for u in tqdm(t["username"]):
        try:
            res[u] = model.docvecs[u]
        except:
            ec += 1
    print("{}/{} users do not have d2v emb".format(ec, t.shape[0]))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--col-name", required=True, dest="col_name")

    args = parser.parse_args()
    start = time.time()

    model_path = f"models/{args.col_name}_doc2vec.model"
    start = time.time()
    logging.info("loading corpus...")
    train_corpus = load_corpus(args.col_name)
    logging.info("training model...")
    model = train_doc2vec(train_corpus, col=args.col_name)
    logging.info("saving model...")
    model.save(model_path)

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

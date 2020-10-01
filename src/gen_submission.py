#!/usr/bin/env python

__author__ = "Tam Nguyen"
__email__ = "tam@aidatech.io"


import argparse
import logging
import pandas as pd
import numpy as np


def main(input_predict_file, submission_file):
    pred = np.loadtxt(input_predict_file)
    pred = np.where(pred < 0, 0, pred)
    np.savetxt(submission_file, np.round(pred), fmt="%i")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-predict-file", required=True, dest="input_predict_file"
    )
    parser.add_argument("--submission-file", required=True, dest="submission_file")

    args = parser.parse_args()
    start = time.time()
    main(
        args.input_predict_file, args.submission_file,
    )

    logging.info("finished ({:.2f} sec elapsed)".format(time.time() - start))

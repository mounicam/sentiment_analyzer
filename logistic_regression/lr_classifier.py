#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import argparse
import numpy as np
from lr import FeatureExtractor 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


def main(args):
    fext = FeatureExtractor()
    train_feats, train_labels, train_tweets = fext.get_features(args.train)
    test_feats, test_labels, test_tweets = fext.get_features(args.test)

    clf = LogisticRegression()
    clf.fit(train_feats, train_labels)
    predicted = clf.predict(test_feats)
    
    pickle.dump(clf, open(args.model, 'wb'))

    print(classification_report(test_labels, predicted))
    print("Acc", accuracy_score(test_labels, predicted))
    print("F1", f1_score(test_labels, predicted, average="macro"))
    print("Recall", recall_score(test_labels, predicted, average="macro"))
    print("Precision", precision_score(test_labels, predicted, average="macro"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--model")
    args = parser.parse_args()
    main(args)

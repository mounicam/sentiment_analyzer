#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


POS_EMOTICONS = {":-)", ":)", ":D", ":o)", ":]", "D:3", ":c)", ":>", "=]", "8)", "=)", ":}", ":ˆ)", ":-D", "8-D",
                 "8D", "xD", "xD", "X-D", "XD", "=-D", "=D", "=-3", "=3", "BˆD", ":')", ":')", ":*", ":-*", ":ˆ*",
                 ";-)", ";)", "*-)", "*)", ";-]", ";]",  ";D", ";ˆ)", ">:P", ":-P", ":P", "X-P", "x-p", "xp", "XP",
                 ":-p", ":p", "=p", ":-b", ":b"}

NEG_EMOTICONS = { ">:[", ":-(", ":(", ":-c", ":-<", ":<", ":-[", ":[", ":{", ";(", ":-||", ">:(",
                  ":'-(", ":'(", "D:<", "D=", "v.v"}

RESOURCES = {
    "isol_pos": "data/isol/positivas_mejorada.csv",
    "isol_neg": "data/isol/negativas_mejorada.csv",
    "embed": "data/embeddings/glove-sbwc.i25.vec"
}


class FeatureExtractor:
    def __init__(self):
        self.isol_pos = set([line.strip().lower() for line in open(RESOURCES["isol_pos"], encoding="ISO-8859-1")])
        self.isol_neg = set([line.strip().lower() for line in open(RESOURCES["isol_neg"], encoding="ISO-8859-1")])

        self.embeddings = {}
        i = 0
        for line in open(RESOURCES["embed"]):
            tokens = line.strip().split()
            self.embeddings[tokens[0].strip().lower()] = [float(tok) for tok in tokens[1:]]

            if i % 100000 == 0:
                print(i)
            i += 1
        print("Done Loading", len(self.embeddings))

    def get_features(self, fname):
        features, labels, tweets = [], [], []
        for line in open(fname):

            tokens = line.strip().split("\t")
            
            if len(tokens) < 2:
                tokens.append(" ")

            label = tokens[0]
            tweet = tokens[1]

            labels.append(label.strip())

            tweet_comb = "".join(tweet.split())
            fv = list()
            fv.append(sum([token in self.isol_pos for token in tweet.lower().split()]))
            fv.append(sum([token in self.isol_neg for token in tweet.lower().split()]))

            pos_emo = sum([token in tweet_comb for token in POS_EMOTICONS])
            neg_emo = sum([token in tweet_comb for token in NEG_EMOTICONS])
            fv.append(pos_emo)
            fv.append(neg_emo)

            vectors = [self.embeddings[tok] for tok in tweet.lower().split() if tok in self.embeddings]
            if len(vectors) > 0:
                fv.extend(np.mean(vectors, axis=0).tolist())
            else:
                fv.extend([0] * 300)

            features.append(fv)
            tweets.append(tweet)

        print(len(features), len(features[0]), len(labels), len(tweets))
        return features, labels, tweets


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

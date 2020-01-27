import os
import math
import pickle
import json
import ftfy
import argparse
import numpy as np
import product_of_experts as poe
from nltk.tokenize import TweetTokenizer
from logistic_regression.lr import FeatureExtractor


TMP_PYTEXT_INPUT = "/tmp/pytext_input.tsv"
TMP_PYTEXT_OUTPUT = "/tmp/test_out.txt"
LABELS = {0: "positive", 1: "negative", 2: "neutral"}


def cnn_classifier(model):
    os.system("pytext test --no-cuda --model-snapshot {} --test-path {}".format(model, TMP_PYTEXT_INPUT))
    return poe.cnn_classifier(TMP_PYTEXT_OUTPUT)


def process_spanish_tweets(fname, cnn_model, lg_model):

    print("Loading Spanish tweets")
    
    # Write to an input format
    i = 0
    spanish_tweets = {}
    for line in open(fname):
        tweet_dict = json.loads(line.strip())
        tweet_text = tweet_dict['text_m']
        tweet_text = ftfy.fix_text(tweet_text)
        tweet_text = tweet_text.replace("\n", " ")
        if tweet_dict["lang"] == "es" and len(tweet_text) > 0:
            spanish_tweets[i] = tweet_text

        if i % 50000 == 0:
            print(str(len(spanish_tweets)) + " out of " + str(i) + " Tweets are Spanish")
        i += 1

    print("Done loading Spanish tweets")

    tknzr = TweetTokenizer()

    fp = open(TMP_PYTEXT_INPUT, "w")
    for k, v in spanish_tweets.items():
        tokens = tknzr.tokenize(v)
                
        final_tokens = []
        for tok in tokens:
            if not tok.startswith("@") and not tok.startswith("http"):
                final_tokens.append(tok)
            if tok.startswith('#') and len(tok[1:]) > 0:
                final_tokens.append(tok[1:])

        fp.write("0" + "\t" + " ".join(final_tokens) + "\t" + str(k) + "\n")
    fp.close()

    print("Classifying Spanish tweets")

    # Do classification
    cnn_preds, cnn_scores = cnn_classifier(cnn_model)
    lr_preds, lr_scores, _ = poe.lr_classifier(lg_model, TMP_PYTEXT_INPUT)
    input_lines = open(TMP_PYTEXT_INPUT).readlines()

    assert len(lr_scores) == len(cnn_scores)
    assert len(cnn_preds) == len(lr_preds)

    # Combine labels across ensembles
    for line, cnn_pred, cnn_score, lr_pred, lr_score in zip(input_lines, cnn_preds, cnn_scores, lr_preds, lr_scores):
        _, tweet, index = line.strip().split("\t")
        index = int(index)
        assert index in spanish_tweets

        final_score, final_label = poe.product_of_experts(lr_score, cnn_score)
        final_label_dict = LABELS[final_label]
        final_score_dict = dict()
        final_score_dict["positive"] = final_score[0]
        final_score_dict["negative"] = final_score[1]
        final_score_dict["neutral"] = final_score[2]

        org_tweet = spanish_tweets[index]
        spanish_tweets[index] = (org_tweet, final_label_dict, final_score_dict)

        # print(tweet)
        # print(org_tweet)
        # print(cnn_pred, cnn_score, lr_pred, lr_score, final_score, final_score_dict, final_label, final_label_dict)

        assert 0.99 < sum(lr_score) < 1.01 and 0.99 < sum(cnn_score) < 1.01

    print("Classification DONE")

    return spanish_tweets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lg")
    parser.add_argument("--cnn")
    parser.add_argument("--input")
    args = parser.parse_args()
    process_spanish_tweets(args.input, args.cnn, args.lg)


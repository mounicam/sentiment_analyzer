import os
import math
import pickle
import argparse
import numpy as np
from logistic_regression.lr import FeatureExtractor
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


"""
Final classifier for Spanish sentiment analysis on TASS test sets.
"""


def cnn_classifier(fname):
    pytext_out_lines = [line.strip() for line in open(fname)][1:]

    scores = []
    predicted = []
    for py_out in pytext_out_lines:
        py_out = py_out.strip().split("\t")
        label = py_out[0]
        predicted.append(label)
        score = [math.exp(float(tok)) for tok in py_out[2][1:-1].split(",")]

        # Order of labels is different
        score = [score[2], score[0], score[1]]

        scores.append(score)

    assert len(predicted) == len(scores)
    return predicted, scores


def lr_classifier(model, fname):
    fext = FeatureExtractor()
    test_feats, test_labels, _ = fext.get_features(fname)
    clf_lr = pickle.load(open(model, 'rb'))
    scores = clf_lr.predict_proba(test_feats)
    predicted = clf_lr.predict(test_feats)
    assert len(predicted) == len(scores)
    return predicted, scores, test_labels


def product_of_experts(lr_score, cnn_score):
    lr_score = np.array(lr_score)
    cnn_score_re = np.array(cnn_score)

    final_score = lr_score * cnn_score_re
    final_score = final_score / sum(final_score)
    final_label = np.argmax(final_score)

    return final_score, final_label


def ensemble(cnn_input, lg_input, lg_model):
    
    # Do classification
    cnn_preds, cnn_scores = cnn_classifier(cnn_input)
    lr_preds, lr_scores, gold = lr_classifier(lg_model, lg_input)

    assert len(cnn_preds) == len(lr_preds)
    assert len(cnn_scores) == len(lr_scores)

    # Combine labels across ensembles
    predicted = []
    for cnn_pred, cnn_score, lr_pred, lr_score in zip(cnn_preds, cnn_scores, lr_preds, lr_scores):
        final_score, final_label = product_of_experts(lr_score, cnn_score)
        predicted.append(final_label)

        print(cnn_pred, cnn_score, lr_pred, lr_score, final_score, final_label)
        assert 0.99 < sum(lr_score) < 1.01 and 0.99 < sum(cnn_score) < 1.01

    gold = [int(i) for i in gold]
    print(classification_report(gold, predicted))
    print("Acc", accuracy_score(gold, predicted))
    print("F1", f1_score(gold, predicted, average="macro"))
    print("Recall", recall_score(gold, predicted, average="macro"))
    print("Precision", precision_score(gold, predicted, average="macro"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lg_model", help='Pretrained logistic regression model.\n')
    parser.add_argument("--cnn_input", help='Predictions of CNN from Pytext.\n')
    parser.add_argument("--lg_input", help="Path to test set")
    args = parser.parse_args()
    ensemble(args.cnn_input, args.lg_input, args.lg_model)


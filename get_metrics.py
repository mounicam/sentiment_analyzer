import argparse
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

"""
Calculate metrics using TASS test set.
Example output of CNN classifier is in pretrained_models.
"""


def print_metrics(fname):

    gold = []
    predicted = []
    lines = open(fname).readlines()[1:]
    for line in lines:
        if len(line.strip()):
            line = line.strip().split("\t")
            gold.append(line[1])
            predicted.append(line[0])

    print(classification_report(gold, predicted))
    print("Acc", accuracy_score(gold, predicted))
    print("F1", f1_score(gold, predicted, average="macro"))
    print("Recall", recall_score(gold, predicted, average="macro"))
    print("Precision", precision_score(gold, predicted, average="macro"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()

    print_metrics(args.input)

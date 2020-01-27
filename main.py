import sys
sys.path.append("tweetment")

import json
import time
import ftfy
import argparse
import tweetment
from spanish_sentiment_analysis import process_spanish_tweets


class Config:
    def __init__(self):
        self.spanish_lg = "pretrained_models/lr.pkl"
        self.spanish_cnn = "pretrained_models/model.pt"
        self.english_svm = "tweetment/model_ntiez.pkl"


def main(args):

    start = time.time()

    config = Config()

    # All the spanish tweets are extracted and processed
    spanish_predictions = process_spanish_tweets(args.input, config.spanish_cnn, config.spanish_lg)

    # Classifier for English sentiment analysis
    english_classifier = tweetment.SentimentClassifier(config.english_svm)

    f_out = open(args.out, "w")

    i = 0
    for line in open(args.input):
        tweet_dict = json.loads(line.strip())
        tweet_text = tweet_dict['text_m']
        tweet_text = ftfy.fix_text(tweet_text)
        tweet_text = tweet_text.replace("\n", " ")

        if tweet_dict["lang"] == "en":
            (pred_label, scores) = english_classifier.classify(tweet_text)
        elif tweet_dict["lang"] == "es" and len(tweet_text) > 0:
            assert i in spanish_predictions
            tweet, pred_label, scores = spanish_predictions[i]
            assert tweet_text == tweet
        else:
            pred_label = "N/A"
            scores = "N/A"

        # print(tweet_text, tweet_dict["lang"],pred_label, scores)

        if 'extension' not in tweet_dict:
            tweet_dict['extension'] = json.loads("{}")

        scores_str = json.dumps(scores)
        tweet_dict['extension']['predicted_sentiment'] = pred_label
        tweet_dict['extension']['sentiment_scores'] = scores_str
        tweet_dict_str = json.dumps(tweet_dict) + "\n"
        f_out.write(tweet_dict_str)
        f_out.flush()

        if i % 5000 == 0:
            end = time.time()
            print("time_elapsed: ", end - start, "secs")
            print("processed: " + str(i))
        i += 1

    f_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help='Input json file.\n')
    parser.add_argument("--out", help='Output json file.\n')
    args = parser.parse_args()
    main(args)

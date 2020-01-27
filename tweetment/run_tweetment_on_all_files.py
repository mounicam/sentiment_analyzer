import datetime
import json
from os import listdir
from os.path import isfile, join
import gzip
import sys

import time
import ftfy
import glob
from nltk.tokenize import TweetTokenizer

start = time.time()

import tweetment
classifier = tweetment.SentimentClassifier("model_ntiez.pkl")


def process_file(input_file, output_file):

    tknzr = TweetTokenizer()

    count_processed_tweet = 0
    
    with gzip.open(input_file,'r') as fin, gzip.open(output_file, 'wb') as f_out:
        for line in fin:        
            tweet_dict=json.loads(line.strip())
            tweet_text = tweet_dict['text_m']
            tweet_text=ftfy.fix_text(tweet_text)
            tweet_dict['text_m']=tweet_text
            tweet_text = " ".join(tknzr.tokenize(tweet_text.replace("\n", " ")))

            if tweet_dict["lang"] == "en":

                (pred_label, all_label_scores) = classifier.classify(tweet_text)
                all_label_scores_str = json.dumps(all_label_scores)

                tweet_dict['predicted_sentiment']=pred_label
                tweet_dict['sentiment_scores']=all_label_scores_str

                print count_processed_tweet, tweet_text, pred_label, all_label_scores, tweet_dict["lang"]

                tweet_dict_str = json.dumps(tweet_dict)+"\n"
                f_out.write(tweet_dict_str)
                f_out.flush()

            if count_processed_tweet == 1000:
                end = time.time()
                print("time_elapsed: ",end - start, "secs")
                print("processed: "+str(count_processed_tweet))
                break
            count_processed_tweet += 1


if __name__ == '__main__':
    # input_dir_name = "/home/maddela/other_projects/socialsim/July_30/raw_input/twitter/"
    # output_dir_name="/home/maddela/other_projects/socialsim/July_30/twitter/"
    #
    # files = glob.glob(input_dir_name)
    #
    # for input_file in files:
    #
    #     output_file = input_file.split("/")[-1].replace(".json.gz","-sent-analysis.json.gz")
    #     output_file = output_dir_name + output_file
    #     print(input_file, output_file)

        process_file("/scratch/maddela/socialsim-20200118/CP4_Ven/cp4-ven_twitter_text_4OSU-nlp.json.gz", "test1.json.gz")

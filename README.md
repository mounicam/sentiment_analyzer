# Multilingual Sentiment Analyzer


## Steps to run the code: 
1.  Install pytext. (Python version > 3.6)
```
  $ python3 -m venv pytext_venv
  $ source pytext_venv/bin/activate
  (pytext_venv) $ cd pytext 
  (pytext_venv) pip install .
```

2. Download Spanish glove embeddings to data/embeddings directory. You can find the embeddings [here](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz). Unzip the file.


 
3. Run the main file. It takes a json file as an input and assumes that 'text_m' is the field with tweet text. It adds a json object called 'extension' for each tweet with predicted sentiment and sentiment scores. 
 
```
  $ source pytext_venv/bin/activate
  $ (pytext_venv) python3 main.py --input <input json file> --out <output json file>
```


## Repo structure

1. ```data```: Contains isol lexicon (for logistic regression classifier), tass dataset (to train Spanish sentiment analyzer) and Spanish glove embeddings.

2. ```logistic_regression```: Code for logistic regression classifier for Spanish sentiment analysis

3. ```pretrained_models```: Pretrained models (logisitic regression and CNN) for Spanish sentiment analysis

4. ```pytext```: Code for CNN classfier for Spanish sentiment analysis

5. ```tweetment```: Code for SVM classfier for English sentiment analysis

6. ```main.py```: Code for multi-lingual sentiment analyzer (English and Spanish). 


## References:

1. For English sentiment analysis, we used [tweetment](https://github.com/ntietz/tweetment) library.

2. For Spanish sentiment analysis, we implemented this [paper](https://www.aclweb.org/anthology/E17-1095.pdf). 

# import all necessary libraries
import json
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords, stem_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim

df_filter = pd.read_excel('/Users/itoshiki/Documents/nlp_lab/general/filter.xlsx')

data = df_filter['Radiology text']

def clean_txt(txt):
    new_txt = txt.replace('\n',' ')
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords, stem_text]
    words = preprocessing.preprocess_string(txt.lower(), CUSTOM_FILTERS)
    return words

new_data = list()
for d in data:
    new_l = clean_txt(str(d))
    new_data.append(new_l)


# keep the same as the tfidf dimension
EMBEDDING_DIM = 300

model = gensim.models.Word2Vec(sentences=new_data,
        size=EMBEDDING_DIM,
        window=5,
        workers=4,
        min_count=1)

filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

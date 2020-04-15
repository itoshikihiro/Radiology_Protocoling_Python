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
import logging

dv = gensim.models.doc2vec.Doc2Vec.load("../data/doc2vec_trained.model")

# load data as dataframe
df_filter = pd.read_excel('../data/tune.xlsx')

# filter all data without any empty data
df_filter = df_filter.fillna('N/A')
df_filter = df_filter[df_filter['Radiology text']!='N/A']


def cpt_ext(txt):
    try:
        splited_list = txt.lower().split('\n\n')
        new_txt = splited_list[0]+" "+splited_list[1]
        new_txt = new_txt.replace('\n', ' ')
        return new_txt
    except:
        return 'N/A'


def clean_txt(txt):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords, stem_text]
    words = preprocessing.preprocess_string(txt.lower(), CUSTOM_FILTERS)
    if not words:
        return 'N/A'
    return words


# data cleaning for CPT
def df_clean_CPT(df_filter):
    # general cleaning empty entry
    df_return = df_filter.fillna('N/A')
    df_return = df_return[df_return['Radiology text']!='N/A']
    
    # specific cleaning empty entry in CPT_text
    # empty entries mean failed convertion during the extraction process
    df_return['CPT_text'] = df_return['Radiology text'].apply(cpt_ext)
    df_return = df_return[df_return['CPT_text']!='N/A']
    # transferring words to sentences
    df_return['CPT_text'] = df_return['CPT_text'].apply(clean_txt)
    df_return = df_return[df_return['CPT_text']!='N/A']
    return df_return

df_filter = df_clean_CPT(df_filter)

def vectorize_data(dv, data):
    return np.vstack([dv.infer_vector(d) for d in data])

X= vectorize_data(dv, df_filter['CPT_text'])
y= df_filter['cpt_label']

clf = LogisticRegression(random_state=0, class_weight='balanced')

def log_reg_test_2():
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],        
        'solver': ['liblinear', 'saga'],
   }
    search = GridSearchCV(clf,param_grid,n_jobs=-1,scoring = 'accuracy')
    search = search.fit(X, y)
    print(search.best_params_)
    print(search.best_score_)


log_reg_test_2()

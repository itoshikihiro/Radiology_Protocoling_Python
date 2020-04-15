
# coding: utf-8

# In[1]:


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
from sklearn.metrics import roc_auc_score
from string import ascii_uppercase

import gensim
import logging

import re



# load pre-trained word2vec model
wv = gensim.models.KeyedVectors.load_word2vec_format("../../data/embedding_word2vec.txt", binary=False)
wv.init_sims(replace=True)


# In[3]:


def remove_special_char(txt):
    return re.sub(r'[^a-zA-Z0-9 :,_/;.]',r'',txt)


# In[3]:


def remove_empty(df_filter, filter_name):
    return_val = df_filter.copy()
    return_val = return_val.fillna('N/A')
    return_val = return_val[return_val[filter_name]!='N/A']
    return return_val


# In[4]:


def remove_unreadable(txt):
    return re.sub(r'_[a-zA-Z0-9]+_',r'\n',txt)


# In[5]:
def format_str(txt):
    return_val = txt.replace('\r',' ')
    return_val = return_val.strip()
    return_val = re.sub(r'(\s*\n\s*){2,}',r';', return_val)
    return_val = return_val.replace('(\n)+',' ')
    return_val = re.sub(r'(\s)+',r' ', return_val)
    return_val = return_val.strip()
    return return_val


# In[6]:


def preprocess_radi_txt(df_filter, filter_str):
    return_val = df_filter.copy()
    # remove empty entries
    return_val = remove_empty(return_val, filter_str)
    # remove unreadable str
    # remove speical char
    return_val[filter_str] =  return_val[filter_str].apply(remove_unreadable)
    # remove empty entries
    return_val = remove_empty(return_val, filter_str)
    # format_str
    return_val[filter_str] =  return_val[filter_str].apply(format_str)
    # remove empty entries
    return_val = remove_empty(return_val, filter_str)
    # format_str
    return_val[filter_str] =  return_val[filter_str].apply(remove_special_char)
    # remove empty entries
    return_val = remove_empty(return_val, filter_str)

    return return_val


# In[9]:


def clean_txt(txt):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords, stem_text]
    words = preprocessing.preprocess_string(txt.lower(), CUSTOM_FILTERS)
    new_words = []
    for word in words:
        word_val = word
        word_val = re.sub(r'([a-zA-Z])+\d+([a-zA-Z])+',r' ', word_val)
        word_val = re.sub(r'([a-zA-Z])+\d+',r' ', word_val)
        word_val = re.sub(r'\d+([a-zA-Z])+',r' ', word_val)
        if word_val == " ":
            new_words = []
            break;
        new_words.append(word)
    if not new_words:
        return 'N/A'
    return new_words


# In[10]:


def cpt_ext(txt):
    try:
        splited_list = txt.split(';')
        new_txt = ""
        for i in splited_list:
            for j in i.split(':'):
                if j.strip().lower()=='procedure':
                    new_txt=i.split(':')[1].strip()
                    return new_txt
                elif j.strip().lower()=='exam':
                    new_txt=i.split(':')[1].strip()
                    return new_txt
        for i in range(len(splited_list)):
            tmp_txt = splited_list[i].strip()
            if 'REPORT'== tmp_txt.split(" ")[0].strip():
                new_txt = tmp_txt.split(" ")[1:]
                new_txt = " ".join(new_txt).strip()
                if new_txt == "":
                    new_txt = splited_list[i+1].strip()
        return new_txt
    except:
        return 'N/A'


# In[11]:


# data cleaning for CPT
def df_clean_CPT(df_filter):
    df_return = df_filter.copy()
    # specific cleaning empty entry in CPT_text
    # empty entries mean failed convertion during the extraction process
    df_return['CPT_text'] = df_return['Radiology text'].apply(cpt_ext)
    df_return = remove_empty(df_return, 'CPT_text')
    # transferring words to sentences
    df_return['CPT_text'] = df_return['CPT_text'].apply(clean_txt)
    df_return = remove_empty(df_return, 'CPT_text')
    return df_return


# In[12]:
def read_df_fr_path(file_path):
    df_filter = pd.read_excel(file_path)
    # filter all data without any empty data
    df_filter = preprocess_radi_txt(df_filter, 'Radiology text')
    return df_filter


def load_data(filepath):
    # load data as dataframe
    df_filter = read_df_fr_path(filepath)
    # filter all data without any empty data
    df_return = df_clean_CPT(df_filter)
    return df_return



# In[7]:


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        print(words)
        return np.zeros(wv.vector_size,)

    mean = np.array(mean).mean(axis=0)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


# In[8]:


train_df = load_data('../../data/train.xlsx')
test_df = load_data('../../data/test.xlsx')


# In[10]:


X_train= word_averaging_list(wv,train_df['CPT_text'])
y_train = train_df['cpt_label']


# In[11]:


X_test= word_averaging_list(wv,test_df['CPT_text'])
y_test = test_df['cpt_label']

# In[16]:


def log_reg():
    print(" ")
    print("logistic regression")
    reg = LogisticRegression(
        random_state=0,
        solver = 'liblinear',
        C = 10,
        penalty = 'l1',
        class_weight='balanced')
    reg = reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print (classification_report(y_test, pred))


# In[ ]:


def ran_for():
    print(" ")
    print("random forest")
    reg = RandomForestClassifier(
        random_state=0,
        max_depth = 12,
        max_features = 50,
        min_samples_split = 2,
        n_estimators = 150,
        class_weight = 'balanced',
        n_jobs=-1)
    reg = reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print (classification_report(y_test, pred))


# In[ ]:


def xgboost_test():
    print(" ")
    print("xgboost")
    reg = XGBClassifier(
        n_estimators = 200,
        max_depth = 11,
        max_features = 50,
        min_samples_split = 2,
        random_state=0,
        n_jobs = -1)
    reg = reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print (classification_report(y_test, pred))


# In[ ]:


log_reg()
ran_for()
xgboost_test()


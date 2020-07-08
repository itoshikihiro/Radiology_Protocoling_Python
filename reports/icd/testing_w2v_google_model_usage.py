#!/usr/bin/env python
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
from sklearn.pipeline import Pipeline
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder

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
import pickle

wv = gensim.models.KeyedVectors.load_word2vec_format("../vectorization_models/GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)


def remove_special_char(txt):
    return re.sub(r'[^a-zA-Z0-9 :,_/;.]',r'',txt)


def remove_empty(df_filter, filter_name):
    return_val = df_filter.copy()
    return_val = return_val.fillna('N/A')
    return_val = return_val[return_val[filter_name]!='N/A']
    return return_val


def remove_unreadable(txt):
    return re.sub(r'_[a-zA-Z0-9]+_',r'\n',txt)


def format_str(txt):
    return_val = txt.replace('\r',' ')
    return_val = return_val.strip()
    return_val = re.sub(r'(\s*\n\s*){2,}',r';;;', return_val)
    return_val = return_val.replace('(\n)+',' ')
    return_val = re.sub(r'(\s)+',r' ', return_val)
    return_val = return_val.strip()
    return return_val


def enhance_formatting(txt):
    new_str = ""
    for tmp_str in txt.split(';;;'):
        if ":" in tmp_str:
            new_str += ";;;"
        new_str += tmp_str.strip() +" "
    return new_str


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
    # format_str
    return_val[filter_str] =  return_val[filter_str].apply(enhance_formatting)
    # remove empty entries
    return_val = remove_empty(return_val, filter_str)
    return return_val


def clean_txt(txt):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords]
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


def icd_ext(txt):
    try:
        splited_list = txt.split(';;;')
        new_txt = ""
        for i in splited_list:
            if ':' in i:
                i_split_list = i.split(':')
                prefix = i_split_list[0].lower()
                if 'impression' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
                if 'history' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
                if 'indication' in prefix:
                    new_txt += i_split_list[1].lower() + '; '
        return new_txt.strip()
    except:
        return 'N/A'



def read_df_fr_path(file_path):
    df_filter = pd.read_excel(file_path)
    # filter all data without any empty data
    df_filter = preprocess_radi_txt(df_filter, 'Radiology text')
    return df_filter



# In[8]:


# data cleaning for CPT
def df_clean_ICD(df_filter):
    df_return = df_filter.copy()
    # specific cleaning empty entry in ICD_text
    # empty entries mean failed convertion during the extraction process
    df_return['ICD_text'] = df_return['Radiology text'].apply(icd_ext)
    df_return = remove_empty(df_return, 'ICD_text')
    # transferring words to sentences
    df_return['ICD_text'] = df_return['ICD_text'].apply(clean_txt)
    df_return = remove_empty(df_return, 'ICD_text')
    return df_return


# In[9]:


icd_10_all = {}
for l in ascii_uppercase:
    for i in range(0,10):
        for j in range(0,10):
            new_str = l+str(i)+str(j)
            if (l=='A') or (l=='B'):
                icd_10_all.update({new_str:'A00-B99'})
            elif (l=='C'):
                icd_10_all.update({new_str:'C00-D49'})
                if (i==4) and (j==4):
                    icd_10_all.update({'C4A':'C00-D49'})
                if (i==7) and (j==7):
                    icd_10_all.update({'C7A':'C00-D49'})
                    icd_10_all.update({'C7B':'C00-D49'})
            elif (l=='D'):
                if (i<=4):
                    icd_10_all.update({new_str:'C00-D49'})
                else:
                    icd_10_all.update({new_str:'D50-D89'})
                if (i==9) and (j==9):
                    icd_10_all.update({'D3A':'C00-D49'})
            elif (l=='E'):
                icd_10_all.update({new_str:'E00-E89'})
            elif (l=='F'):
                icd_10_all.update({new_str:'F01-F99'})
            elif (l=='G'):
                icd_10_all.update({new_str:'G00-G99'})
            elif (l=='H'):
                if (i<=5):
                    icd_10_all.update({new_str:'H00-H59'})
                else:
                    icd_10_all.update({new_str:'H60-H95'})
            elif (l=='I'):
                icd_10_all.update({new_str:'I00-I99'})
            elif (l=='J'):
                icd_10_all.update({new_str:'J00-J99'})
            elif (l=='K'):
                icd_10_all.update({new_str:'K00-K95'})
            elif (l=='L'):
                icd_10_all.update({new_str:'L00-L99'})
            elif (l=='M'):
                icd_10_all.update({new_str:'M00-M99'})
                if (i==1) and (j==4):
                    icd_10_all.update({'M1A':'M00-M99'})
            elif (l=='N'):
                icd_10_all.update({new_str:'N00-N99'})
            elif (l=='O'):
                icd_10_all.update({new_str:'O00-O9A'})
                if (i==9) and (j==9):
                    icd_10_all.update({'O9A':'O00-O9A'})
            elif (l=='P'):
                icd_10_all.update({new_str:'P00-P96'})
            elif (l=='Q'):
                icd_10_all.update({new_str:'Q00-Q99'})
            elif (l=='R'):
                icd_10_all.update({new_str:'R00-R99'})
            elif (l=='S') or (l=='T'):
                icd_10_all.update({new_str:'S00-T88'})
            elif (l=='V') or (l=='W') or (l=='X') or (l=='Y'):
                icd_10_all.update({new_str:'V00-Y99'})
            elif (l=='Z'):
                icd_10_all.update({new_str:'Z00-Z99'})


# In[10]:


def load_data(filepath):
    # load data as dataframe
    df_filter = read_df_fr_path(filepath)
    # filter all data without any empty data
    df_return = df_clean_ICD(df_filter)
    return df_return


# In[11]:


def load_label(df):
    general_icd_label = []
    for i in df['icd_label']:
        splitted_list = i.split('.')
        code = splitted_list[0]
        x = icd_10_all.get(splitted_list[0])
        general_icd_label.append(x)
    return general_icd_label


# In[12]:
data_df = load_data("../data/filter.xlsx")
data_df['general_icd_label'] = load_label(data_df)

features = data_df['ICD_text']
labels = data_df['general_icd_label']

label_encoder = LabelEncoder()
label_encoder.fit(labels)

test_df = load_data("../data/test.xlsx")


test_df['general_icd_label'] = load_label(test_df)


X_test = test_df['ICD_text']
y_test = test_df['general_icd_label']
y_test = label_encoder.transform(y_test)


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

X_test = word_averaging_list(wv, X_test)

# In[9]:


def load_model_predict(model_name, test_data):
    loaded_model = pickle.load(open(model_name, 'rb'))
    result = loaded_model.predict(test_data)
    return result


# In[10]:


predicted_icd = load_model_predict('save_log.sav', X_test)


# In[16]:


new_df = {
    'radiology_text':test_df['Radiology text'],
    'icd_text':test_df['ICD_text'],
    'icd':test_df['icd'],
    'ori_icd_label':label_encoder.inverse_transform(y_test),
    'prediced_icd_label':label_encoder.inverse_transform(predicted_icd)
}


# In[18]:


new_df = pd.DataFrame(new_df)


# In[12]:


new_df.to_csv('test_log.csv')


# In[32]:


f = open("test_log.txt", "w")
f.write(classification_report(y_test, predicted_icd, target_names=label_encoder.classes_))
f.close()


# In[25]:


predicted_icd = load_model_predict('save_ran_for.sav', X_test)


# In[26]:


new_df = {
    'radiology_text':test_df['Radiology text'],
    'icd_text':test_df['ICD_text'],
    'icd':test_df['icd'],
    'ori_icd_label':label_encoder.inverse_transform(y_test),
    'prediced_icd_label':label_encoder.inverse_transform(predicted_icd)
}


# In[27]:


new_df = pd.DataFrame(new_df)


# In[12]:


new_df.to_csv('test_ran_for.csv')


# In[ ]:


f = open("test_ran_for.txt", "w")
f.write(classification_report(y_test, predicted_icd, target_names=label_encoder.classes_))
f.close()


# In[28]:


predicted_icd = load_model_predict('save_xgboost.sav', X_test)


# In[29]:


new_df = {
    'radiology_text':test_df['Radiology text'],
    'icd_text':test_df['ICD_text'],
    'icd':test_df['icd'],
    'ori_icd_label':label_encoder.inverse_transform(y_test),
    'prediced_icd_label':label_encoder.inverse_transform(predicted_icd)
}


# In[30]:


new_df = pd.DataFrame(new_df)


# In[12]:


new_df.to_csv('test_xgboost.csv')


# In[ ]:


f = open("test_xgboost.txt", "w")
f.write(classification_report(y_test, predicted_icd, target_names=label_encoder.classes_))
f.close()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[2]:


from sklearn.datasets import make_classification
from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords, stem_text
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
from string import ascii_uppercase
from sklearn.utils import class_weight

import gensim
import logging

import re
import pickle

from tensorflow.keras.models import model_from_json


# In[3]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[4]:


tf.debugging.set_log_device_placement(True)


# In[5]:


# load word2vec model
wv = gensim.models.KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)


# In[6]:


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


# In[8]:


# load all datasets
data_df = load_data("../data/filter.xlsx")
data_df['general_icd_label'] = load_label(data_df)
features = data_df['ICD_text']
labels = data_df['general_icd_label']


# In[9]:


# to unify the label encoding mechanism
label_encoder = LabelEncoder()
label_encoder.fit(labels)


# In[10]:


# load training dataset
train_df = load_data("../data/train.xlsx")
train_df['general_icd_label'] = load_label(train_df)
X_train = train_df['ICD_text']
y_train = train_df['general_icd_label']


# In[11]:


y_train = label_encoder.transform(y_train)


# In[12]:


# load validation datasets
vali_df = load_data("../data/tune.xlsx")
vali_df['general_icd_label'] = load_label(vali_df)
X_vali = vali_df['ICD_text']
y_vali = vali_df['general_icd_label']


# In[13]:


y_vali = label_encoder.transform(y_vali)


# In[14]:


MAXLEN = 1000


# In[15]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(features)


# In[16]:


vocab_size = len(t.word_index) + 1


# In[17]:


train_sequences = t.texts_to_sequences(X_train)
# re-encode the datasets
train_features = pad_sequences(train_sequences, maxlen=MAXLEN)
train_labels = to_categorical(y_train)


# In[18]:


vali_sequences = t.texts_to_sequences(X_vali)
# re-encode the datasets
vali_features = pad_sequences(vali_sequences, maxlen=MAXLEN)
vali_labels = to_categorical(y_vali)


# In[19]:


# create embedding matrix for embeding layer
# transferring word2vec model to a dictionary
embeddings_index = {}

for key in wv.vocab:
    coefs = np.asarray(wv[key], dtype='float32')
    embeddings_index[key] = coefs


# In[20]:


# the number of 300 here is the actual length of each document
embedding_matrix = np.zeros((vocab_size, 300))


# In[21]:


for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[23]:


train_labels.shape


# In[24]:


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAXLEN,), dtype='int32'),
    tf.keras.layers.Embedding(vocab_size,
                              300,
                              weights=[embedding_matrix],
                              input_length=MAXLEN,
                              trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='softmax'),
    tf.keras.layers.Dense(train_labels.shape[1])
])


# In[25]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])


# In[26]:


model.summary()


# In[27]:


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=20, verbose=1, baseline=None, restore_best_weights=True)


# In[28]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(list(labels) ),
                                                 list(labels))


# In[29]:


history = model.fit(train_features , train_labels,
                    epochs=100, verbose=True, shuffle = True,
                    validation_data = (vali_features,vali_labels), class_weight=class_weights,
                    batch_size=64, callbacks=[es])


# In[ ]:


with open('./RNNtrainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# serialize model to JSON
model_json = model.to_json()
with open("./model_RNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./model_RNN.h5")
print("Saved RNN model to disk")


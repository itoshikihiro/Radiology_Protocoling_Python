#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)


# In[3]:


# load word2vec model
wv = gensim.models.KeyedVectors.load_word2vec_format("../../data/GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)


# In[4]:


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


# In[5]:


# load all datasets
data_df = load_data("../../data/filter.xlsx")


# In[6]:


features= data_df['CPT_text']
labels = data_df['cpt_label']


# In[7]:


# to unify the label encoding mechanism
label_encoder = LabelEncoder()
label_encoder.fit(labels)


# In[8]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(features)

vocab_size = len(t.word_index) + 1


# In[9]:


train_df = load_data("../../data/train.xlsx")
X_train = train_df['CPT_text']
y_train = train_df['cpt_label']


# In[10]:


# load validation datasets
vali_df = load_data("../../data/tune.xlsx")
X_vali = vali_df['CPT_text']
y_vali = vali_df['cpt_label']


# In[11]:


y_train = label_encoder.transform(y_train)
y_vali = label_encoder.transform(y_vali)


# In[12]:


MAXLEN = 1000


# In[13]:


train_sequences = t.texts_to_sequences(X_train)
# re-encode the datasets
train_features = pad_sequences(train_sequences, maxlen=MAXLEN)

# validation features re-encoding
vali_sequences = t.texts_to_sequences(X_vali)
# re-encode the datasets
vali_features = pad_sequences(vali_sequences, maxlen=MAXLEN)


# In[14]:


train_labels = to_categorical(y_train)
vali_labels = to_categorical(y_vali)


# In[15]:


# create embedding matrix for embeding layer
# transferring word2vec model to a dictionary
embeddings_index = {}

for key in wv.vocab:
    coefs = np.asarray(wv[key], dtype='float32')
    embeddings_index[key] = coefs

# the number of 300 here is the actual length of each document
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[16]:


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


# In[17]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])


# In[18]:


model.summary()


# In[19]:


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=20, verbose=1, baseline=None, restore_best_weights=True)


# In[20]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(list(labels) ),
                                                 list(labels))


# In[22]:


with tf.device('/CPU:0'):
    history = model.fit(train_features , train_labels,
                        epochs=10000, verbose=True, shuffle = True,
                        validation_data = (vali_features,vali_labels), class_weight=class_weights,
                        batch_size=64, callbacks=[es])


# In[ ]:



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


# In[ ]:





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
import sklearn.preprocessing

import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from string import ascii_uppercase


# load data as dataframe
df_filter = pd.read_excel('../data/tune.xlsx')

# filter all data without any empty data
df_filter = df_filter.fillna('N/A')
df_filter = df_filter[df_filter['Radiology text']!='N/A']


def icd_ext(txt):
    try:
        splited_list = txt.lower().split('impression:')
        new_txt = splited_list[1]
        new_txt = new_txt.replace('\n', ' ')
        return new_txt
    except:
        return 'N/A'


def clean_txt(txt):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,remove_stopwords, stem_text]
    words = preprocessing.preprocess_string(txt.lower(), CUSTOM_FILTERS)
    return ' '.join(words)


# data cleaning for CPT
def df_clean_ICD(df_filter):
    # general cleaning empty entry
    df_return = df_filter.fillna('N/A')
    df_return = df_return[df_return['Radiology text']!='N/A']
    
    # specific cleaning empty entry in CPT_text
    # empty entries mean failed convertion during the extraction process
    df_return['ICD_text'] = df_return['Radiology text'].apply(icd_ext)
    df_return = df_return[df_return['ICD_text']!='N/A']
    # transferring words to sentences
    df_return['ICD_text'] = df_return['ICD_text'].apply(clean_txt)
    return df_return



df_filter = df_clean_ICD(df_filter)


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


general_icd_label = []
for i in df_filter['icd_label']:
    splitted_list = i.split('.')
    code = splitted_list[0]
    x = icd_10_all.get(splitted_list[0])
    general_icd_label.append(x)


df_filter['general_icd_label'] = general_icd_label


X= df_filter['ICD_text']
y= df_filter['general_icd_label']
encoder = sklearn.preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

vect = CountVectorizer(stop_words='english')
tifidf = TfidfTransformer()
clf = XGBClassifier(
        random_state=0,
        n_jobs=-1)
text_clf = Pipeline([('vect', vect),('tfidf', tifidf),('clf', clf)])

def xgboost_test_2():
    param_grid = {
        'vect__ngram_range': [(1,1),(1,2),(1,3)],
        'vect__max_features': range(200, 401, 50),
        'clf__n_estimators': range(50,201,50),
        'clf__max_depth':range(3,14,2),
        'clf__max_features':range(50,201,50),
        'clf__min_samples_split':range(2, 20, 4)
    }
    search = GridSearchCV(text_clf,param_grid,scoring = 'accuracy')
    search = search.fit(X, y)
    print(search.best_params_)
    print(search.best_score_)
    
    
xgboost_test_2()

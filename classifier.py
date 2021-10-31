#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re
import nltk
import pickle
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import xgboost as xgb

df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

df_true = df_true[['text']]
df_fake = df_fake[['text']]

df_true['label'] = 1
df_fake['label'] = 0
df = pd.concat([df_true, df_fake])
df.rename(columns = {'text': 'message'}, inplace= True)

stop = nltk.corpus.stopwords.words('english')

def remove_stopwords(text, is_lower_case=False):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop]

    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

df['message'] = df['message'].str.replace('\d+', '')
df['message'] = df['message'].apply(remove_stopwords)

X = df.message
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)

tv = TfidfVectorizer()
X_train_tv = tv.fit_transform(X_train)
X_test_tv = tv.transform(X_test)

xgb_cl = xgb.XGBClassifier(random_state = 42)
xgb_cl.fit(X_train_tv, y_train)

def binary_classification_performance(y_test, y_pred, model):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall) 
    auc_roc = round(roc_auc_score(y_score = y_pred, y_true = y_test),2)
    model_name = model
    result = pd.DataFrame({
                         'Model' : [model_name],
                         'Precision' : [precision],
                         'Recall': [recall],
                         'f1 score' : [f1_score],
                         'AUC_ROC' : [auc_roc],
                         'True Positive' : [tp],
                         'True Negative' : [tn],
                         'False Positive':[fp],
                         'False Negative':[fn]
                        })
    
    return result


y_pred = xgb_cl.predict(X_test_tv)
binary_classification_performance(y_test, y_pred, xgb_cl)

pickle.dump(xgb_cl, open('model.pkl', 'wb'))
pickle.dump(tv,open('tv.pkl', 'wb'))

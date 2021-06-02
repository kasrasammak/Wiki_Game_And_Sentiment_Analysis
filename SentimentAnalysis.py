#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:08:11 2021

@author: owlthekasra
"""


import pandas as pd



#%%
df = pd.read_csv('Sentiment_Analysis_Dataset.csv', encoding="ISO-8859-1")
df = df.iloc[0::100, :]
df = df[["Sentiment", "SentimentSource","SentimentText"]].reset_index().iloc[:,1:]
# df = df[["Sentiment", "SentimentSource","SentimentText"]]
#%% Separate dataframe into positive and negative sentiment observations
positive= df[df["Sentiment"]==1]["SentimentText"]
negative = df[df["Sentiment"]==0]["SentimentText"]

#%% Remove stop words to find meaningful word frequencies
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
added_words = ["-", "get", "going", "go", "I'm", "im", "u","know", "&amp;", "got", "I'll", "@", "that's", "like", "really", "one", "...", "..", "2", "?", "&lt;3","see"]
stop = list(stop_words)
stop.extend(added_words)
stop = set(stop)
#%% functions to get determine frequency of words
def wordListToFreqDf(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return pd.DataFrame(list(zip(wordlist,wordfreq)))

def getMostFrequent(posneg):
    temp = [wrd for sub in posneg for wrd in sub.split()]
    filt = [w for w in temp if not w.lower() in stop]
    freqdict = wordListToFreqDf(filt)
    show = freqdict.sort_values(by=1,ascending=False)
    o = show.groupby(0).count().sort_values(by=1, ascending=False)
    return o

#%% find most positive words
pos_freq = getMostFrequent(positive[0:2000]).reset_index()
neg_freq = getMostFrequent(negative[0:2000]).reset_index()

pos_freq = pos_freq[pos_freq[0]!="I'm"].reset_index().iloc[:,1:]
neg_freq = neg_freq[neg_freq[0]!="I'm"].reset_index().iloc[:,1:]

top_10_pos = pos_freq.iloc[0:10,0:2]
top_10_neg = neg_freq.iloc[0:10,0:2]

#%%
top_10_pos['rank'] = top_10_pos.index + 1
top_10_pos['zipf'] = top_10_pos[1] * top_10_pos['rank']

top_10_neg['rank'] = top_10_neg.index + 1
top_10_neg['zipf'] = top_10_neg[1] * top_10_neg['rank']
# when multiplying the count by the rank, 
# you do not approach a constant
# and that might be due to removing stop words
#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(top_10_pos[1], top_10_pos['rank'])
sns.lineplot(top_10_neg[1], top_10_neg['rank'])

# The inverse proportionality part of zipf law is true, 
# as you can see a vaguely 1/x to -x graph in the plot 

#%% Machine Learning Classification

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
#%% Getting rid of symbols from text

## from https://stackabuse.com/text-classification-with-python-and-scikit-learn

documents = []

from nltk.stem import WordNetLemmatizer
import re

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

#%% 
X, y = df['SentimentText'], df['Sentiment']

#%%
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stop)
X = tfidfconverter.fit_transform(documents).toarray()

#%%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.1, random_state = 42)

#%% Model Selection
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 
#%%
y_pred = classifier.predict(X_test)
#%%
clf = MultinomialNB().fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

#%%
acc_random_forest = accuracy_score(y_pred, y_test) 
acc_Naive_Bayes = accuracy_score(y_pred_clf, y_test)

#%%
def getKappa(test, pred):
    cm = confusion_matrix(test, pred)
    num = 0
    denom = 0
    obs = 0
    for i in range(0,len(cm)):
        num = num + (sum(cm[i])*sum(cm[:,i]))
        denom = denom+sum(cm[i])
        obs = obs + cm[i,i]
    expected = num/denom
    kappa = (obs - expected)/(denom - expected)
    return kappa

#%%
    kappa_forest = getKappa(y_test,y_pred)
    kappa_np = getKappa(y_test,y_pred_clf)
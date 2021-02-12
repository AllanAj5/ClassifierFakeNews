# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:21:57 2021

@author: Adrien
"""

import pandas as pd
df = pd.read_csv("C:/Users/Adrien/.spyder-py3/Allan Python Files/NLP/ClassifierFakeNews/train.csv")
#test_data = pd.read_csv("test.csv")

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords




# Data Cleaning / Preprocessing
X = df.dropna()  # Droping Rows with NAN
X.reset_index(inplace=True)
y = X.iloc[:,-1]   # alternatively y = df['label']
X = X.drop('label',axis=1)

corpus = []
p_stemmer = PorterStemmer()

import re

for sent in range(0,len(X)):
    review = X['title'][sent].lower()
    review = re.sub("[^a-z]", " ", review)
    review = review.split()
    review = [p_stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Vectorization
from sklearn.feature_extraction.text import  CountVectorizer
bow_vec = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = bow_vec.fit_transform(corpus).toarray()

# Visualize Bow
v = pd.DataFrame(X,columns=bow_vec.get_feature_names())


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state= 0)


# Model Training 
from sklearn.naive_bayes import MultinomialNB
mnb_model = MultinomialNB()
mnb_model.fit(X_train,y_train)

# Testing Our Model
y_pred = mnb_model.predict(X_test)
#Confusion Matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


# Checking for Accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

C = confusion_matrix(y_test,y_pred)
A = accuracy_score(y_test,y_pred)
C_lass = classification_report(y_test,y_pred)


print("Accuracy Score = ",A*100,"%")

# Hyper Param Tuning
prev_score = 0
best_alpha = 0

for alphaS in np.arange(0,1,0.1):
    sub_classifier = MultinomialNB(alpha = alphaS)
    sub_classifier.fit(X_train,y_train)
    sub_y_pred = sub_classifier.predict(X_test)
    score = accuracy_score(y_test,sub_y_pred)
    
    if score>prev_score:
        prev_score = score
        mnb_model = sub_classifier
        print("alpha = {0}, Score = {1}".format(alphaS,score))
        best_alpha = alphaS
        

mnb_model = MultinomialNB(best_alpha)
mnb_model.fit(X_train,y_train)

pred_y = mnb_model.predict(X_test)

acc_racy = accuracy_score(y_test,pred_y)
print("Best Accuracy = ",acc_racy*100,"%")
    




                


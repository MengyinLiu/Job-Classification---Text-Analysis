# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:36:48 2015

@author: Tina Liu
"""

##Salary prediction#####
import nltk
import pandas as pd
import os
import random
os.getcwd()
os.chdir("C:\Tina Liu\Text Analytics\Assignment1\Redo")
from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


train=pd.read_csv("Train_rev1.csv")
job_des=train.ix[:,2]

#tokenize and preprocess the first 5000 posts
text=list()
for job in job_des[:5000]:
    temp = nltk.word_tokenize(job.strip().lower().translate(None,  "\\!#$%&()*+,-./:;<=>?@[\]^_`{|}~"))
    #remove stopwords in job description
    temp2 = [w for w in temp if not w in stop]
    #lemmatize the word
    temp3=list()
    for w in temp2:
        w=repr(w).replace("\\","")
        temp3.append(wordnet_lemmatizer.lemmatize(w))
    text.append(temp3)
text_series=pd.Series(text)

#salary for each post
salaries = train.ix[:4999,10]
salaries_sort=sorted(salaries)
percentile = .75*len(salaries_sort)
seven_five_percentile = salaries_sort[int(percentile)]
salary_class=["low"]*5000
for i in range(len(salary_class)):
    if salaries[i]>=seven_five_percentile:
        salary_class[i]="high"
    
#pick out the most frequent 5000 words 
job_tuple=[]     
for i in range(len(text_series)):
  job_tuple.append((text_series[i],salary_class[i]))
random.shuffle(job_tuple)

all_text=[]
for text in text_series:
    all_text.extend(text)
    
   
all_words = nltk.FreqDist(w.lower() for w in all_text )
word_features = list(all_words)[:5000]

#create a function to decide whether a word is in featured words or not
def document_features(document): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

 	
featuresets = [(document_features(d), c) for (d,c) in job_tuple]
train_set, test_set = featuresets[1500:], featuresets[:1500]
classifier = nltk.NaiveBayesClassifier.train(train_set)


print(nltk.classify.accuracy(classifier, test_set))

#show most imformative words
informative=classifier.show_most_informative_features(50)

#Confusion matrix setup
testoutput=[]
for (jobdesc,label) in test_set:
  a = classifier.classify(jobdesc)
  testoutput.append(a)
actual=[]
for (jobdesc,label) in test_set:
  actual.append(label)

#Confusion matrix
cm = nltk.ConfusionMatrix(actual, testoutput)
print(cm.pp(sort_by_count=True, show_percents=True, truncate=2))






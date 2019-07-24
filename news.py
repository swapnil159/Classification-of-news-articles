# Predicting category of news article

# importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from time import time
import re
import sys
import csv


# importing dataset
sys.path.append("../home/swapnil/Desktop/Assignment/Section3")
data = pd.read_csv('news.csv',error_bad_lines=False)
#print(data.info())

# Encoding CATEGORY
encoder = LabelEncoder()
test = encoder.fit_transform(data['CATEGORY'])
#print(y[:5])

# Cleaning the Titles
print("Cleaning Titles")
t = time()
sw = stopwords.words("english")
ss = SnowballStemmer("english")	
corpus = []
for i in range(0,422419):
	headline = re.sub("[^a-zA-Z]",' ',data['TITLE'][i])					## Removing punctuation marks
	headline = headline.lower()								## Converting the titles to lowercase
	headline = headline.split()								## Spliting the titles into words
	headline = [ss.stem(word) for word in headline if not word in set(sw)]			## Removing stopwords and then stemming the words
	headline = ' '.join(headline)								## Concatenating the words 
	corpus.append(headline)									## Modified List of Titles
print("Time Taken: ",round(time()-t),"s")

# Creating Bag of Words model
transformer = CountVectorizer(stop_words="english",max_features=1000)
train = transformer.fit_transform(corpus).toarray()
#print(transformer.get_feature_names())

# Partioning the dataset
features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(train,test,test_size=0.2,random_state=0)
#print(features_train.info())

# Fitting using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)							## Accuracy: 0.877

# Fitting using Naive Bayes
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()										## Accuracy: 0.824

# Fitting using Decision Trees
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()								## Accuracy: 0.860

# Fitting from Logistic Regression
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0,multi_class='multinomial')				## Accuracy: 0.872

## Tried SVM as well but kept on getting memory error for 1000 features
## It ran on 500 features giving an accuracy of 0.78



t = time()
print("Training...")
clf.fit(features_train,labels_train)
print("Time Taken: ",round(time()-t),"s")

acc = clf.score(features_test,labels_test)
print("Using Logistic Regression: ",acc)

print("Predicting...")
t = time()
result = clf.predict(features_test)
print("Time Taken: ",round(time()-t),"s")

print(confusion_matrix(labels_test,result))
print(classification_report(labels_test, result))

#from scipy.stats import chisquare
#score, pval = chisquare(result, labels_test)
#print(score)
#print(pval)
results_cv = cross_val_score(clf, train, test, cv=10)
print(results_cv.mean())



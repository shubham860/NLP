import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 

from nltk.corpus import stopwords  #Removing stop words like emojies for he she etc..
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re

dataset = pd.read_csv('train.csv')

processed_tweets = []

dataset['tweet'][0]

for i in range(31962):
    tweet = re.sub('@[\W]*' , ' ' , dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z#]', ' ' , tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    
    tweet = [ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)    
    processed_tweets.append(tweet)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(processed_tweets)
X  = X.toarray()

y = dataset['label'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb.score(X_train,y_train)


print(cv.get_feature_names())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

dataset = pd.read_csv('At&T_Data.csv')

processed_title = []
processed_review = []
dataset['Titles'][0]

for i in range(113):
    title = dataset['Titles'][i].lower()
    title = re.sub('@[\W]*' , ' ' , title)
    title = re.sub('[^a-zA-Z]', ' ' , title)
    title = title.split()
    title = [ps.stem(token) for token in title if not token in set(stopwords.words('english'))]
    title = ' '.join(title) 
    processed_title.append(title)

    review = re.sub('@[\W]*' , ' ' , dataset['Reviews'][i])
    review = re.sub('[^a-zA-Z]', ' ' , review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(token) for token in review if not token in set(stopwords.words('english'))]
    review = ' '.join(review) 
    processed_review.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X1 = cv.fit_transform(processed_title)
X2 = cv.fit_transform(processed_review)
X1 = X1.toarray()
X2 = X2.toarray()

y = dataset['Label'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X1_train,X1_test,y_train,y_test = train_test_split(X1,y)
X2_train,X2_test,y_train,y_test = train_test_split(X2,y)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X1_train,y_train)
nb.score(X1_train,y_train)
nb.score(X1_test,y_test)

nb.fit(X2_train,y_train)
nb.score(X2_train,y_train)
nb.score(X2_test,y_test)


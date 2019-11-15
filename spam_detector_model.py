import pandas as pd
import numpy as np
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv('SMSSpamCollection',sep='\t'
	,names = ['label','message'])

x,y = df['message'],df['label']

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)

clf = MultinomialNB()
clf.fit(x_train,y_train)

print(clf.score(x_test,y_test))

to_predict = [['Hello customer u won 10000$.'],['Credit card form']
	,['Hello hope u r doin well']]

for i in to_predict:
	predict_i = cv.transform(i).toarray()
	print(clf.predict(predict_i))

pickle.dump(cv,open('CountVectorizer.pkl','wb'))
pickle.dump(clf,open('MultinomialNB.pkl','wb'))
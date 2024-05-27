import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle

cause = ['Caught in or between', 'Other', 'Fall (from elevation)', 'Struck-by', 'Card-vascular/resp. fail.', 'Shock', 'Struck against', 'Inhalation',
         'Fall (same level)', 'Absorption', 'Rubbed/abraded', 'Bite/sting/scratch', 'Rep. Motion/pressure', 'Ingestion']

frame = pd.read_csv('dataset/test.csv')
frame = frame.dropna()

frame['Event type'] = frame['Event type'].replace('Caught in or between',0)
frame['Event type'] = frame['Event type'].replace('Other',1)
frame['Event type'] = frame['Event type'].replace('Fall (from elevation)',2)
frame['Event type'] = frame['Event type'].replace('Struck-by',3)
frame['Event type'] = frame['Event type'].replace('Card-vascular/resp. fail.',4)
frame['Event type'] = frame['Event type'].replace('Shock',5)
frame['Event type'] = frame['Event type'].replace('Struck against',6)
frame['Event type'] = frame['Event type'].replace('Inhalation',7)
frame['Event type'] = frame['Event type'].replace('Fall (same level)',8)
frame['Event type'] = frame['Event type'].replace('Absorption',9)
frame['Event type'] = frame['Event type'].replace('Rubbed/abraded',10)
frame['Event type'] = frame['Event type'].replace('Bite/sting/scratch',11)
frame['Event type'] = frame['Event type'].replace('Rep. Motion/pressure',12)
frame['Event type'] = frame['Event type'].replace('Ingestion',13)


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

cv = CountVectorizer(analyzer=process_text,stop_words = "english", lowercase = True)
messages_bow = cv.fit_transform(frame['Abstract Text'])
X_train, X_test, y_train, y_test = train_test_split(messages_bow, frame['Event type'], test_size = 0.20, random_state = 0)

cpickle.dump(cv.vocabulary_,open("model/feature.pkl","wb"))
cpickle.dump(X_train,open("model/xtrain.pkl","wb"))
cpickle.dump(X_test,open("model/xtest.pkl","wb"))
cpickle.dump(y_train,open("model/ytrain.pkl","wb"))
cpickle.dump(y_test,open("model/ytest.pkl","wb"))



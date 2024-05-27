import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
from sklearn import svm
from sklearn.metrics import accuracy_score
msg = 'killed while standing beside a conveyor when he was struck by a fallen'
cvv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
cv1 = CountVectorizer(vocabulary=cvv.get_feature_names(),stop_words = "english", lowercase = True)
test = cv1.fit_transform([msg])

X_train = cpickle.load(open("model/xtrain.pkl", "rb"))
X_test = cpickle.load(open("model/xtest.pkl", "rb"))
y_train = cpickle.load(open("model/ytrain.pkl", "rb"))
y_test = cpickle.load(open("model/ytest.pkl", "rb"))
print(X_train.shape)

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    accuracy = accuracy_score(y_test,y_pred)*100
    return accuracy



cls = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
cls.fit(X_train.toarray(), y_train) 
prediction_data = prediction(X_test.toarray(), cls) 
svm_acc = cal_accuracy(y_test, prediction_data)
print(svm_acc)

cvv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
cv1 = CountVectorizer(vocabulary=cvv.get_feature_names(),stop_words = "english", lowercase = True)
test = cv1.fit_transform([msg])
print('Predicted value: ',cls.predict(test.toarray()))
                

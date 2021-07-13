# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:42:27 2018
@author: kbera
"""

import numpy as np
import pandas as pd
import math

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFdr, SelectFpr, \
    GenericUnivariateSelect, SelectPercentile
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from xgboost import XGBClassifier

# f_classif = 0.67, chi2 = 0.67, mutual = 0.67
def selectKBest(fin, test, K, score_func):
    # fin = SelectKBest(chi2, k=4).fit_transform(X.values, y.values)
    # print(f_classif(fin.iloc[:, 1:].values, fin.iloc[:, :1].values)[0])
    selector = SelectKBest(score_func, k=K).fit(fin.iloc[:, 1:].values, fin.iloc[:, :1].values.flatten())
    scores = selector.scores_.flatten()
    # x_new = selector.transform(X.values)
    # print(scores)
    # plot_feature_scores(scores, 0)
    ind = np.argpartition(scores, -K)[-K:]  # Get MAX. 4 features
    # print(type(ind))
    # print(fin[0])
    ind = ind + 1
    ind = np.append(0, ind)
    fin = fin[fin.columns[ind]]
    test = test[test.columns[ind]]
    # print(fin.head())
    # print(test.head())
    return fin, test


def predictor(model, fin, test):
    model.fit(fin.iloc[:, 1:], fin.iloc[:, :1].values.flatten())
    predicted = model.predict(test.iloc[:, 1:])
    return predicted


def findAccuracy(test, predicted):
    tn, fp, fn, tp = confusion_matrix(test.iloc[:, :1], predicted).ravel()
    print("tn:", tn, "fp", fp, "fn:", fn, "tp:", tp)
    # print("rate of positive: ", (tp + fp) / (tp + fp + tn + fn))
    # print(sum(test.iloc[:, :1].values) / len(test))
    print("accuracy:", accuracy_score(test.iloc[:, :1], predicted))
    print("recall:", tp/(tp + fn))
    print("precision:", tp/(tp + fp))
    return


def shuffler(model):
    model = shuffle(model)
    # model.to_csv("./dataset/shuffled_reg.csv", index=False, encoding="utf-8")
    return model

def plot_bar_chart(time, not_time):
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
     
    objects = ('Delayed', 'On time')
    y_pos = np.arange(len(objects))
    performance = [time, not_time]
     
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Sample Count')
    plt.title('Delayed vs Non-delayed')
     
    plt.show()

def plot_feature_scores(scores, score_func_index):
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    
    score_func_str = ['ANOVA F-value', 'mutual information', 'chi-square']

    score_list = []
    for score in scores:
        print(int(score))
        score_list.append(score)

    objects = ('departure humidity' ,'departure pressure', 'departure temperature', 'departure wind', 'arrival humidity', 'arrival pressure', 'arrival temperature', 'arrival wind')
    y_pos = np.arange(len(objects))
    performance = score_list

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Score')
    plt.title(str(score_func_str[score_func_index]) + 'Feature Scores')
    plt.show()

# model = pd.read_csv("./dataset/final.csv")
model = pd.read_csv("./dataset/final.csv")

# SHUFFLING DATASET
model = shuffler(model)
model = model.dropna()

model = model.drop('dep_desc', 1)
model = model.drop('arr_desc', 1)
test = model[:10000]
model = model[10000:]
threshold = 20


train_ones = model[(model['DEPARTURE_DELAY'] >= threshold) | (model['ARRIVAL_DELAY'] >= threshold)]
train_zeros = model[(model['DEPARTURE_DELAY'] < threshold)  & (model['ARRIVAL_DELAY'] < threshold)]

#plot_bar_chart(train_ones.shape[0],train_zeros.shape[0])
train_ones = train_ones.sort_values('DEPARTURE_DELAY', ascending=False)
train_zeros = train_zeros.sort_values('DEPARTURE_DELAY', ascending=False)

train_ones = train_ones[:200000]
train_zeros = train_zeros[:200000]

fin = pd.concat([train_ones, train_zeros], axis=0, ignore_index=True)
fin.DEPARTURE_DELAY[(fin.DEPARTURE_DELAY < threshold) & (fin.ARRIVAL_DELAY < threshold)] = 0
fin.DEPARTURE_DELAY[(fin.DEPARTURE_DELAY >= threshold) | (fin.ARRIVAL_DELAY >= threshold)] = 1

fin = fin.drop('ARRIVAL_DELAY', 1)

test.DEPARTURE_DELAY[(test.DEPARTURE_DELAY < threshold) & (test.ARRIVAL_DELAY < threshold)] = 0
test.DEPARTURE_DELAY[(test.DEPARTURE_DELAY >= threshold) | (test.ARRIVAL_DELAY >= threshold)] = 1

test = test.drop('ARRIVAL_DELAY', 1)

score_func = [f_classif, mutual_info_classif, chi2]
mdl = LogisticRegression(class_weight = 'balanced')
#mdl = GaussianNB()
#mdl = MultinomialNB()
for sc in range(len(score_func)):
    print("score_func: ", sc)
#    for i in range(8):
#    K = i+1
    K = 8
    print("K is: ", K)
    fin_copy, test_copy = selectKBest(fin, test, K, score_func[sc])
    predicted = predictor(mdl, fin_copy, test_copy)
    findAccuracy(test_copy, predicted)
    
## SELECTING K-BEST FEATURES
#K = 8
#fin, test = selectKBest(fin, test, K)
#
##mdl = GaussianNB()
## mdl = SGDClassifier()  # worst predictions
#mdl = LogisticRegression()
##mdl = svm.SVC(gamma="scale", class_weight="balanced")
##mdl = LinearDiscriminantAnalysis()
##mdl = XGBClassifier()
#predicted = predictor(mdl, fin, test)
#
#findAccuracy(test, predicted)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
import os.path
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix

def analysis():
    datainput = pd.read_csv("C:/Users/dayan/OneDrive/Documents/Final Project/CAD/Source Code/CAD_dataset.csv")
    y = datainput['class']
    del datainput['class']
    X_train, X_test, y_train, y_test = train_test_split(datainput, y, test_size=0.2, random_state=42)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    hyperparameters = {
        'n_estimators': 100,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
    }

    rfc_clf =RandomForestClassifier(**hyperparameters)



    rfc_clf.fit(X_train, y_train)

    predicted = rfc_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, average='macro') * 100

    recall = recall_score(y_test, predicted, average='macro') * 100

    fscore = f1_score(y_test, predicted, average='macro') * 100

    print("RF=", accuracy, precision, recall, fscore)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    predicted = knn_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, average='macro') * 100

    recall = recall_score(y_test, predicted, average='macro') * 100

    fscore = f1_score(y_test, predicted, average='macro') * 100

    print("KNN=", accuracy, precision, recall, fscore)


    xgb_clf = ExtraTreesClassifier(max_depth=4, random_state=42)

    xgb_clf.fit(X_train, y_train)

    predicted = xgb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, average='macro') * 100

    recall = recall_score(y_test, predicted, average='macro') * 100

    fscore = f1_score(y_test, predicted, average='macro') * 100

    print("XBoost=", accuracy, precision, recall, fscore)



    rfc_clff = RandomForestClassifier(max_depth=3, random_state=42)

    knn_clff = KNeighborsClassifier()
    ET_clff = ExtraTreesClassifier(max_depth=4, random_state=42)

    voting_clf = VotingClassifier(estimators=[('RF', rfc_clff), ('ET',ET_clff), ('dt', knn_clff)], voting='hard')

    voting_clf.fit(X_train, y_train)

    predicted = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, pos_label=1) * 100

    recall = recall_score(y_test, predicted, pos_label=1) * 100

    fscore = f1_score(y_test, predicted, pos_label=1) * 100


    print("VTC=", accuracy, precision, recall, fscore)


analysis()







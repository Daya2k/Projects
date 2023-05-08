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
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix

def analysis():
    datainput = pd.read_csv("C:/Users/dayan/OneDrive/Documents/Final Project/atrial_fibriilation/atrial_fibrillation_model.csv")
    datainput = datainput.head(100000)
    y = datainput['ritmi']
    del datainput['ritmi']
    X_train, X_test, y_train, y_test = train_test_split(datainput, y, test_size=0.25, random_state=542)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    hyperparameters = {
        'n_estimators': 100,
        'max_depth': 35,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'class_weight': {0:1, 1:5},
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

    print("Classifier Accuracy Precision Recall F-Score")
    print("RF:        ", round(accuracy, 2),'  ',round(precision, 2),'  ',round(recall, 2),'  ',round(fscore, 2))


    xgb_clf = XGBClassifier(n_estimators=100, max_depth=35, learning_rate=0.05)

    xgb_clf.fit(X_train, y_train)

    predicted = xgb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, average='macro') * 100

    recall = recall_score(y_test, predicted, average='macro') * 100

    fscore = f1_score(y_test, predicted, average='macro') * 100

    print("XGB:       ", round(accuracy, 2),'  ',round(precision, 2),'  ',round(recall, 2),'  ',round(fscore, 2))


    knn_clf = KNeighborsClassifier(n_neighbors=25)
    knn_clf.fit(X_train, y_train)

    predicted = knn_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, average='macro') * 100

    recall = recall_score(y_test, predicted, average='macro') * 100

    fscore = f1_score(y_test, predicted, average='macro') * 100

    print("KNN:       ", round(accuracy, 2),'  ',round(precision, 2),'  ',round(recall, 2),'  ',round(fscore, 2))



#    rfc_clff = RandomForestClassifier(max_depth=5, random_state=1242)
    ET_clff = ExtraTreesClassifier(max_depth=32, random_state=742, class_weight={0:1,1:5})
    dt_clf = DecisionTreeClassifier(max_depth=32, min_samples_split=2, class_weight={0:1,1:5})

    knn_clff = KNeighborsClassifier()



    voting_clf = VotingClassifier(estimators=[('ET', ET_clff), ('KNN', knn_clf), ('DT', dt_clf)], voting='soft')

    voting_clf.fit(X_train, y_train)

    predicted = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted) * 100

    precision = precision_score(y_test, predicted, pos_label=1, average='macro') * 100

    recall = recall_score(y_test, predicted, pos_label=1, average='macro') * 100

    fscore = f1_score(y_test, predicted, pos_label=1, average='macro') * 100


    print("VTC:       ", round(accuracy, 2),'  ',round(precision, 2),'  ',round(recall, 2),'  ',round(fscore, 2))


analysis()







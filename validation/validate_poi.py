#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]
def classify(features_train, labels_train):

    ### your code goes here--should return a trained decision tree classifer
    X = features_train
    Y = labels_train
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    return clf
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.30, random_state=42)

### it's all yours from here forward!  


clf = classify(features_train, labels_train)
labels_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, labels_pred)
acc
# %load poi_id.py
# !/usr/bin/python

import numpy as np


def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = (group.mean() + stds * group.std())
    return group


import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Abhimanyu- Initially we will try to accomodate all features that are available .
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'percentage_to_poi',
                 'percentage_from_poi']  # You will need to use more features
initial_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                         'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                         'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                         'shared_receipt_with_poi', 'to_messages', 'from_messages', 'from_poi_to_this_person',
                         'from_this_person_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
# The Total data point will not help us in the evaluation
identified_outliers = ["TOTAL"]

print("Original Length", len(data_dict))

for outlier in identified_outliers:
    data_dict.pop(outlier)

keys = data_dict.keys()

print("Length after Outlier", len(data_dict))


### Abhimanyu- Datapoints which are more than a certain standard deviation cause great skew after feature scaling
### Here I have taken four as the standard deviation above which I have truncated the points to curtail skew
### caused by outliers
from sklearn import preprocessing

for f in initial_features_list:
    a = [data_dict[k][f] for k in keys]
    for i in range(0, len(a)):
        if a[i] == "NaN":
            a[i] = 0
    a = np.array(a).astype(np.float)
    # replacing anything more than 4 Std Dev with 4 Std Dev
    a = replace(a, 4)
    # Scaling Data
    ta_scaled = preprocessing.minmax_scale(a)
    i = 0
    for key in keys:
        data_dict[key][f] = ta_scaled[i]
        i = i + 1
### Task 3: Create new feature(s)
### Abhimanyu Now Adding two new features of % of mails sent from poi to this person and % mails sent from a person to poi.
### This is because just an absolute number of emails might not be indicative as person who has longer history in Enron
### might have sent or recieved more emails while might not be a POI

for key in keys:
    if data_dict[key]['from_poi_to_this_person'] != 0:
        data_dict[key]['percentage_from_poi'] = (data_dict[key]['from_poi_to_this_person']) / float(
            data_dict[key]['to_messages'])
    else:
        data_dict[key]['percentage_from_poi'] = 0
    if data_dict[key]['from_this_person_to_poi'] == 0:
        data_dict[key]['percentage_to_poi'] = 0
    else:
        data_dict[key]['percentage_to_poi'] = (data_dict[key]['from_this_person_to_poi']) / float(
            data_dict[key]['from_messages'])
    data_dict[key]['ctc'] = data_dict[key]['salary'] + data_dict[key]['bonus'] + data_dict[key][
        'exercised_stock_options']

my_dataset = data_dict
### The below code/functions transform features into an NP.Array to be processed later and removes NAN. Just preprocesing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### http://scikit-learn.org/stable/modules/pipeline.html

## Provided to give you a starting point. Try a variety of classifiers.
## Trying different classifier like Gaussian Naive Bayes , Support Vector Machine and Decision Tree. Not trying ensemble
## techniques since data only consists of 145 points


from sklearn.naive_bayes import GaussianNB

clfGB = GaussianNB()
from sklearn.svm import SVC

clfSV = SVC(kernel='rbf', C=100)
from sklearn.tree import DecisionTreeClassifier

clfDT = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Abhimanyu Tried splitting training and testing data by using cross validation . Keeping 30% data for Training here
### The purpose is just to pick one classifying technique over the other
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)
#
clfGB.fit(features_train, labels_train)
predGB = clfGB.predict(features_test)
clfSV.fit(features_train, labels_train)
predSV = clfSV.predict(features_test)
clfDT.fit(features_train, labels_train)
predDT = clfDT.predict(features_test)
from sklearn.metrics import classification_report

target_names = ['Not PoI', 'PoI']
print("GaussianNB")
print(classification_report(labels_test, predGB, target_names=target_names))

print("Support Vector Classifier")
print(classification_report(labels_test, predSV, target_names=target_names))

print("Decision Tree")
print(classification_report(labels_test, predDT, target_names=target_names))

### Abhimanyu Manually found that Decision Tree is Giving Best Results. Can remove feature scaling since DT doesnt require feature scaling

### Abhimanyu Tuning the decision tree to best params using Grid Search CV


### Abhimanyu Trying Parameter Tuning to get Best Params
from sklearn.model_selection import GridSearchCV

param_grid = {'min_samples_split': np.arange(2, 10)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(features_train, labels_train)
print(tree.best_params_)

clf = tree
dump_classifier_and_data(clf, my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics


# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
clfDT = DecisionTreeClassifier(min_samples_split=7)
# The "accuracy" scoring is proportional to the number of correct
# classifications
RFECV = RFECV(estimator=clfDT, step=1, cv=StratifiedShuffleSplit(labels, 1000, random_state=42),
              scoring="f1")
RFECV.fit(features, labels)


print("optimal number of features", RFECV.n_features_)
print("Now Printing Features Priority")
for i in range(0, len(RFECV.ranking_)):
    print(features_list[i+1], RFECV.ranking_[i])
print(RFECV.ranking_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (f1 score)")
plt.plot(range(1, len(RFECV.grid_scores_) + 1), RFECV.grid_scores_)
plt.show()

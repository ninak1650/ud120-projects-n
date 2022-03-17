#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import _pickle as pickle
import re

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(len(enron_data))

print(len(enron_data["SKILLING JEFFREY K"]))

count_poi = 0
for i in enron_data:
    if enron_data[i]["poi"] == 1:
        count_poi += 1

print(count_poi)

poi_all = 0
with open("../final_project/poi_names.txt") as f:
    content = f.readlines()
for line in content:
    if re.match(r'\((y|n)\)', line):
        poi_all += 1

print(poi_all)

print(enron_data["PRENTICE JAMES"]["total_stock_value"])

print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

enron_keyPOIPayment = dict((k, enron_data[k]['total_payments']) for k in ("LAY KENNETH L", "SKILLING JEFFREY K",
                                                                          "FASTOW ANDREW S"))
max_earner = max(enron_keyPOIPayment, key=enron_keyPOIPayment.get)
print(max_earner, enron_keyPOIPayment[max_earner])

enron_data.get('SKILLING JEFFREY K', [])

count_quant_salary = 0
count_email_address = 0

for i in enron_data:
    if enron_data[i]["email_address"] != "NaN":
        count_email_address += 1
    if enron_data[i]["salary"] != "NaN":
        count_quant_salary += 1

print(count_quant_salary)

print(count_email_address)

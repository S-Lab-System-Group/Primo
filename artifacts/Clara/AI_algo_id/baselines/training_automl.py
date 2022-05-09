import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pickle
from tpot import TPOTClassifier

with open("../embedding.pickle", "rb") as f:
    data = pickle.load(f)
source_reformatted_train = data[0]
target_reformatted_train = data[1]
source_reformatted_ptest = data[2]
target_reformatted_ptest = data[3]
source_reformatted_ntest = data[4]
target_reformatted_ntest = data[5]
# Create a automl Classifier
neigh = TPOTClassifier(generations=8, population_size=10, cv=2, random_state=42, verbosity=2)
neigh.fit(np.array(source_reformatted_train), np.array(target_reformatted_train))

summation_n = 0
total_n = 0
for item in source_reformatted_ntest[:]:
    total_n += 1
    if neigh.predict(np.array([item])) == [1]:
        # print(neigh.predict([item]))
        summation_n += 1
# print(summation, total)
summation_p = 0
total_p = 0
for item in source_reformatted_ptest[:]:
    total_p += 1
    if neigh.predict(np.array([item])) == [1]:
        summation_p += 1
# print(summation, total)
print("Precision: %.3f" % (float(summation_p) / (summation_n + summation_p)))
print("Recall: %.3f" % (float(summation_p) / total_p))


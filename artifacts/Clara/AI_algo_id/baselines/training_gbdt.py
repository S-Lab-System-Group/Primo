import warnings

warnings.filterwarnings("ignore")
import pickle
from sklearn.ensemble import GradientBoostingClassifier

with open("../embedding.pickle", "rb") as f:
    data = pickle.load(f)
source_reformatted_train = data[0]
target_reformatted_train = data[1]
source_reformatted_ptest = data[2]
target_reformatted_ptest = data[3]
source_reformatted_ntest = data[4]
target_reformatted_ntest = data[5]
# Create a gbdt Classifier
neigh = GradientBoostingClassifier(n_estimators=20, random_state=3)
neigh.fit(source_reformatted_train, target_reformatted_train)

summation_n = 0
total_n = 0
for item in source_reformatted_ntest[:]:
    total_n += 1
    if neigh.predict([item]) == [1]:
        summation_n += 1
print(summation_n)
summation_p = 0
total_p = 0
for item in source_reformatted_ptest[:]:
    total_p += 1
    if neigh.predict([item]) == [1]:
        summation_p += 1
# print(summation, total)
print("Precision: %.3f" % (float(summation_p) / (summation_n + summation_p)))
print("Recall: %.3f" % (float(summation_p) / total_p))

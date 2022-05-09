import warnings

warnings.filterwarnings("ignore")
import pickle
from sklearn.neural_network import MLPClassifier

with open("../embedding.pickle", "rb") as f:
    data = pickle.load(f)
source_reformatted_train = data[0]
target_reformatted_train = data[1]
source_reformatted_ptest = data[2]
target_reformatted_ptest = data[3]
source_reformatted_ntest = data[4]
target_reformatted_ntest = data[5]

source_reformatted_train += source_reformatted_train[-40:][:] * 20
target_reformatted_train += target_reformatted_train[-40:][:] * 20
# Create a dnn Classifier
neigh = MLPClassifier(
    solver="sgd",
    activation="relu",
    alpha=1e-4,
    hidden_layer_sizes=(128),
    random_state=1,
    max_iter=10,
    verbose=0,
    learning_rate_init=0.01,
)
neigh.fit(source_reformatted_train, target_reformatted_train)

summation_n = 0
total_n = 0
for item in source_reformatted_ntest[:]:
    total_n += 1
    if neigh.predict([item]) == [1]:
        summation_n += 1
# print(summation, total)
summation_p = 0
total_p = 0
for item in source_reformatted_ptest[:]:
    total_p += 1
    if neigh.predict([item]) == [1]:
        summation_p += 1
# print(summation, total)
print("Precision: %.3f" % (float(summation_p) / (summation_n + summation_p)))
print("Recall: %.3f" % (float(summation_p) / total_p))

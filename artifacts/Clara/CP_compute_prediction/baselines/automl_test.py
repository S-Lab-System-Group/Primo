import warnings

warnings.filterwarnings("ignore")
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--state", help="state")
args = parser.parse_args()
state = int(args.state)

with open("../training.pickle", "rb") as ftrain:
    dataset_train = pickle.load(ftrain)
    X, Y = dataset_train
with open("../testing.pickle", "rb") as ftest:
    dataset_test = pickle.load(ftest)
    X_test, Y_test = dataset_test
with open("../nf.pickle", "rb") as factual:
    dataset_actual = pickle.load(factual)
    X_actual, Y_actual = dataset_actual
with open("../source.pickle", "rb") as fsource:
    source_text_to_int = pickle.load(fsource)
with open("../target.pickle", "rb") as ftarget:
    target_text_to_int = pickle.load(ftarget)
"""
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=7, min_child_weight=18, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.5, verbosity=0)),
    FastICA(tol=0.9),
    ElasticNetCV(l1_ratio=0.8, tol=0.001)
)
"""
"""
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=11, min_samples_split=13, n_estimators=100)),
    KNeighborsRegressor(n_neighbors=7, p=2, weights="uniform")
)
"""
"""
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.0001)),
    XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=8, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.35000000000000003, verbosity=0)
)
"""
exported_pipeline = RandomForestRegressor(
    bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=7, min_samples_split=13, n_estimators=100
)
# Fix random state for all the steps in exported pipeline
# set_param_recursive(exported_pipeline.steps, 'random_state', state)

if hasattr(exported_pipeline, "random_state"):
    setattr(exported_pipeline, "random_state", 42)

exported_pipeline.fit(X, Y)
answer = exported_pipeline.predict(X_actual)

pred = exported_pipeline.predict(X_test)
test_sum = 0
for index in range(len(pred)):
    test_sum += (abs(Y_test[index] * 64 - pred[index] * 64)) / (Y_test[index] * 64)
pred_train = exported_pipeline.predict(X)
train_sum = 0
for index in range(len(pred_train)):
    train_sum += (abs(Y[index] * 64 - pred_train[index] * 64)) / (Y[index] * 64)
print("Train Loss {}".format(train_sum / len(pred_train)))
print("Test Loss {}".format(test_sum / len(pred)))

summation = 0
jndex = 0
pos = 0
nfs = ["aggcounter", "anonipaddr", "forcetcp", "tcp_gen", "tcpack", "tcpresp", "timefilter", "udpipencap"]
len_nfs = [15, 5, 17, 15, 2, 19, 12, 4]
# print(sum(len_nfs), len_nfs, nfs)
nn = a = b = c = 0
temp_list = []

if True:
    for index in range(89):
        a += answer[index]
        b += Y_actual[index]
        c += abs(answer[index] - Y_actual[index])
        summation += abs(answer[index] - Y_actual[index]) / Y_actual[index]
        nn += abs(answer[index] - Y_actual[index]) / Y_actual[index]
        if len_nfs[pos] > 1:
            len_nfs[pos] -= 1
        else:
            temp_var = c / a
            temp_list.append(temp_var[0])
            pos += 1
            a = b = c = nn = 0
    print("Performance on real Click elements: ")
    for index, item in enumerate(temp_list):
        print("WMAPE of:", nfs[index], item)

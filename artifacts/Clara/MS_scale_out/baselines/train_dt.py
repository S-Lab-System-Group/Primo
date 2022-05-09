import pickle
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    with open("dataset.pickle", "rb") as f:
        data = pickle.load(f)

    features_train = data[0]
    tags_train = data[1]
    features_test = data[2]
    tags_test = data[3]
    # print (features_test.shape, tags_test.shape)

    neigh = DecisionTreeClassifier()
    neigh.fit(features_train, tags_train)
    pred = neigh.predict(features_test)
    # print(pred, tags_train)
    pred = list(pred)
    tags_test = list(tags_test)
    # print("prediction: ", pred[:])
    # print("test tags: ", tags_test[:])
    mae = 0
    for i in range(len(pred)):
        mae += abs(pred[i] - tags_test[i])
    print("DT MAE: ", float(mae) / len(pred))


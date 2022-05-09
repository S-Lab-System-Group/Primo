import xgboost as xg
import pickle

if __name__ == "__main__":
    with open("dataset.pickle", "rb") as f:
        data = pickle.load(f)

    features_train = data[0]
    tags_train = data[1]
    features_test = data[2]
    tags_test = data[3]
    # print (features_test.shape, tags_test.shape)
    with open("testset.pickle", "rb") as f:
        data = pickle.load(f)
    features_nf = data[0]
    tags_nf = data[1]
    xgb_r = xg.XGBRegressor(n_estimators=20, seed=42)
    xgb_r.fit(features_train, tags_train)
    pred = xgb_r.predict(features_test)
    # print("feature importance: ", xgb_r.feature_importances_)
    # print(pred, tags_train)
    pred = list(pred)
    tags_test = list(tags_test)
    # print("prediction: ", pred[:])
    # print("test tags: ", tags_test[:])
    mae = 0
    for i in range(len(pred)):
        mae += abs(pred[i] - tags_test[i])
    print("Clara MAE: ", float(mae) / len(pred), "\n")
    with open("testset.pickle", "rb") as f:
        data = pickle.load(f)
    features_nf = data[0]
    tags_nf = data[1]

    pred = xgb_r.predict(features_nf)
    tags_nf = list(tags_nf)
    # print(tags_nf)
    # print(pred)
    print("Performance on Click NFs: ")
    print("MazuNAT optimal: ", tags_nf[-4], ", prediction: ", pred[-4])
    print("DNSProxy optimal: ", tags_nf[-3], ", prediction: ", pred[-3])
    print("UDPCount optimal: ", tags_nf[-2], ", prediction: ", pred[-2])
    print("WebGen optimal: ", tags_nf[-1], ", prediction: ", pred[-1])
    print("")

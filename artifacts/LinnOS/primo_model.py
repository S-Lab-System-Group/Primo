import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from primo.model import PrimoClassifier

feature_names = ["Len_R4", "Len_R3", "Len_R2", "Len_R1", "Len_C", "Lat_R4", "Lat_R3", "Lat_R2", "Lat_R1"]


def main(args):
    drive = args.drive
    trace = f"raw{drive}"
    train_input_path = f"./traces/{trace}.csv"
    cfile = f"primodrive{drive}"

    train_data = pd.read_csv(train_input_path, dtype="float32", sep=",", header=None)
    train_data = train_data.sample(frac=1, random_state=123).reset_index(drop=True)
    train_data = train_data.values

    train_input = train_data[:, :9]
    train_output = train_data[:, 9]

    lat_threshold = np.percentile(train_output, 85)
    print("lat_threshold: ", lat_threshold)
    num_train_entries = int(len(train_output) * 0.80)
    print("num train entries: ", num_train_entries)

    train_Xtrn = train_input[:num_train_entries, :]
    train_Xtst = train_input[num_train_entries:, :]
    train_Xtrn = np.array(train_Xtrn)
    train_Xtst = np.array(train_Xtst)

    # Classification
    train_y = []
    for num in train_output:
        labels = [0] * 2
        if num < lat_threshold:
            labels[0] = 1.0
        else:
            labels[1] = 1.0
        train_y.append(labels)

    train_ytrn = train_y[:num_train_entries]
    train_ytst = train_y[num_train_entries:]
    train_ytrn = np.argmax(train_ytrn, axis=1)
    train_ytst = np.argmax(train_ytst, axis=1)

    """For fast result reprodcution, we disable HPO and model selection. Use specific model type and configuration."""
    dt_config = {
        "prune_factor": 0.001,
    }
    prdt = PrimoClassifier(model="PrDT", model_config=dt_config, hpo=None)
    prdt.fit(train_Xtrn, train_ytrn, train_Xtst, train_ytst)
    train_y_pred = prdt.predict(train_Xtst)
    print(classification_report(train_ytst, train_y_pred, digits=4))

    joblib.dump(prdt.prModel.tree, f"./traces/primodt_{drive}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primo-LinnOS")
    parser.add_argument("-d", "--drive", default=0, type=int, help="Drive ID")

    args = parser.parse_args()
    main(args)

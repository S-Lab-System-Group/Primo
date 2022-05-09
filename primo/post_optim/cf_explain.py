import logging
import numpy as np
from prettytable import PrettyTable
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def find_counterfactual(primo_model, X, y_target, X_refer, y_refer, k=3, distance_type="minkowski"):
    """Counterfactual guide

    TODO: Counterfactual example generation
    """
    # Obtain ebm model
    if "Primo" in type(primo_model).__name__:
        ebm = primo_model.prModel.gam
    elif "PrAM" in type(primo_model).__name__:
        ebm = primo_model.gam
    else:
        raise NotImplementedError

    # Preprocessing
    scaler = MinMaxScaler()
    X_refer = scaler.fit_transform(X_refer)
    X = scaler.transform(X)

    target_idx = np.argwhere(y_refer == y_target).reshape(1, -1)[0]
    X_refer = X_refer[target_idx]

    # Find Counterfactual
    btree = BallTree(X_refer, metric=distance_type)
    dist, idx = btree.query(X, k=k)
    X_cf = X_refer[idx[0]]

    table = PrettyTable(["CF", "Index", "Distance", "CF Value", "Comparsion"])
    for i in range(len(idx[0])):
        diff_index = np.argwhere(X[0] != X_cf[i])
        table.add_row([i + 1, idx[0][i], dist[0][i], X_cf[i], diff_index])

    logger.info(table)
    # logger.info(X)
    # logger.info(X_cf[i])

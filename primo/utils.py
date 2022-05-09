import sys
import os
import logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def logger_init(file):
    logger = logging.getLogger()
    logging.getLogger().handlers.clear()
    logger.setLevel(logging.INFO)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%b %d %H:%M:%S")
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)

    if file is not None:
        handler_file = logging.FileHandler(f"{file}.log", "w")
        handler_file.setLevel(logging.INFO)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    return logger


def score_mapper(score_str):
    score_dict = {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
    }
    assert score_str in score_dict, "Check your `score_fn` value."
    return score_dict[score_str]


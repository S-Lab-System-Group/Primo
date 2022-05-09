import logging
import numpy as np
from abc import ABCMeta
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

from primo.utils import score_mapper

logger = logging.getLogger(__name__)


class BasePrAMModel(BaseEstimator, metaclass=ABCMeta):
    """Base Class for PrAM Models
    """

    @abstractmethod
    def __init__(
        self,
        *,
        score_fn,
        penalty_factor,
        transformer,
        feature_names,
        feature_types,
        max_bins,
        interactions,
        learning_rate,
        seed,
    ):
        self.score_fn = score_fn
        self.penalty_factor = penalty_factor
        self.transformer = transformer
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.interactions = interactions
        self.learning_rate = learning_rate
        self.seed = seed
        self.gam = None

    def fit(self, X, y):
        """ Fits model to provided training dataset.

        Args:
            X: Training samples.
            y: Training labels.
        """

        if is_classifier(self):
            self.gam = ExplainableBoostingClassifier(
                feature_names=self.feature_names,
                feature_types=self.feature_types,
                max_bins=self.max_bins,
                interactions=self.interactions,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )
        else:
            self.gam = ExplainableBoostingRegressor(
                feature_names=self.feature_names,
                feature_types=self.feature_types,
                max_bins=self.max_bins,
                interactions=self.interactions,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )
        self.gam.fit(X, y)
        return self

    def get_model_summary(self):
        return f"PrAM Summary: Shape function number = {len(self.gam.feature_importances_)}"

    def visualize(self, n_features=10, path=None):
        if not path:
            path = "./pram_visualized"

        feature_score = self.gam.feature_importances_
        sorted_score = sorted(feature_score, reverse=True)
        sort_idx = sorted(range(len(feature_score)), key=lambda x: feature_score[x], reverse=True)
        sorted_feature = [self.gam.feature_names[i] for i in sort_idx]

        if n_features > len(feature_score):
            n_features = len(feature_score)

        fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        x = np.arange(1, n_features + 1)
        ax.barh(x[::-1], sorted_score[:n_features], label=sorted_feature[:n_features], alpha=0.8, linewidth=1, edgecolor="k")
        ax.set_yticks(x)
        ax.set_yticklabels(sorted_feature[:n_features][::-1])
        ax.set_xlabel(f"Average Absolute Score")
        fig.savefig(f"{path}.png", bbox_inches="tight")

        logger.info(f'Model visualization file is saved at "{path}.png"')

    def local_visualize(self, X, y, idx=0, n_features=10, path=None):
        if not path:
            path = "./pram_local_visualized"

        explain_class = self.gam.explain_local(X, y)
        localdict = explain_class._internal_obj["specific"]
        intercept = self.gam.intercept_
        features = localdict[idx]["names"]
        scores = localdict[idx]["scores"]

        sorted_score = sorted(scores, key=lambda x: abs(x), reverse=True)
        sort_idx = sorted(range(len(scores)), key=lambda x: abs(scores[x]), reverse=True)
        sorted_feature = [features[i] for i in sort_idx]

        if n_features > len(features):
            n_features = len(features)

        color_list = []
        for i in range(len(scores)):
            if sorted_score[i] >= 0:
                color_list.append("tab:blue")
            else:
                color_list.append("tab:orange")

        fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        x = np.arange(1, n_features + 1)
        ax.barh(
            x[::-1],
            sorted_score[:n_features],
            label=sorted_feature[:n_features],
            alpha=0.8,
            linewidth=1,
            edgecolor="k",
            color=color_list,
        )
        ax.axvline(x=0, ls="-", c="crimson", lw=1.8, clip_on=False)
        ax.set_yticks(x)
        ax.set_yticklabels(sorted_feature[:n_features][::-1])
        ax.set_xlabel(f"Feature Score")
        ax.legend(
            [f"Intercept = {round(intercept, 4)}",]
        )
        fig.savefig(f"{path}.png", bbox_inches="tight")

        logger.info(f'Model local visualization file is saved at "{path}.png"')


class PrAMClassifier(BasePrAMModel, ClassifierMixin):
    """PrAM Classifier
    """

    def __init__(
        self,
        *,
        score_fn="accuracy",
        penalty_factor=0,
        transformer=None,
        feature_names=None,
        feature_types=None,
        max_bins=256,
        interactions=10,
        learning_rate=0.01,
        seed=123,
    ):
        """PrAM Classifier

        Args:
            score_fn (str): Score function. Support most common metrics likes accuracy, f1, mae, mape. Defaults to "accuracy".
            penalty_factor (float): Penalty factor of model scale. The larger value prefer more compact model. Defaults to 0.
            transformer ([type], optional): Apply for Distill Engine for feature transfer. Defaults to None.

            ###### Below keeps same with EBM ######
            feature_names ([type], optional): List of feature names. Defaults to None.
            feature_types ([type], optional): List of feature types. Defaults to None.
            max_bins (int, optional): Max number of bins per feature for pre-processing stage. Defaults to 256.
            interactions (int, optional): Number of interactions of features. Defaults to 10.
            learning_rate (float, optional): Learning rate for boosting. Defaults to 0.01.
            seed (int, optional): Random state. Defaults to 123.
        """

        super(PrAMClassifier, self).__init__(
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            transformer=transformer,
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            interactions=interactions,
            learning_rate=learning_rate,
            seed=seed,
        )

    def score(self, X, y):
        f_score = score_mapper(self.score_fn.lower())
        score = f_score(X, y) + self.penalty_factor / np.sqrt(self.max_bins * self.interactions)
        return score

    def predict(self, X):
        prediction = self.gam.predict(X)
        return prediction


class PrAMRegressor(BasePrAMModel, RegressorMixin):
    """PrAM Regressor
    """

    def __init__(
        self,
        *,
        score_fn="mae",
        penalty_factor=0,
        transformer=None,
        feature_names=None,
        feature_types=None,
        max_bins=256,
        interactions=10,
        learning_rate=0.01,
        seed=123,
    ):
        """PrAM Regressor

        Args:
            score_fn (str): Score function. Support most common metrics likes accuracy, f1, mae, mape. Defaults to "accuracy".
            penalty_factor (float): Penalty factor of model scale. The larger value prefer more compact model. Defaults to 0.
            transformer ([type], optional): Apply for Distill Engine for feature transfer. Defaults to None.

            ###### Below keeps same with EBM ######
            feature_names ([type], optional): List of feature names. Defaults to None.
            feature_types ([type], optional): List of feature types. Defaults to None.
            max_bins (int, optional): Max number of bins per feature for pre-processing stage. Defaults to 256.
            interactions (int, optional): Number of interactions of features. Defaults to 10.
            learning_rate (float, optional): Learning rate for boosting. Defaults to 0.01.
            seed (int, optional): Random state. Defaults to 123.
        """

        super(PrAMRegressor, self).__init__(
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            transformer=transformer,
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            interactions=interactions,
            learning_rate=learning_rate,
            seed=seed,
        )

    def score(self, X, y):
        f_score = score_mapper(self.score_fn.lower())
        score = f_score(X, y) + self.penalty_factor * np.sqrt(self.max_bins * self.interactions)
        return -1 * score

    def predict(self, X):
        prediction = self.gam.predict(X)
        return prediction

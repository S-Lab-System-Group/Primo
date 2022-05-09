import logging
import pydotplus
import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from primo.utils import score_mapper

logger = logging.getLogger(__name__)


class BasePrDTModel(BaseEstimator, metaclass=ABCMeta):
    """Base Class for PrDT Models
    """

    @abstractmethod
    def __init__(
        self, *, score_fn, penalty_factor, prune_factor, transformer, feature_names, seed,
    ):
        self.score_fn = score_fn
        self.penalty_factor = penalty_factor
        self.prune_factor = prune_factor
        self.transformer = transformer
        self.feature_names = feature_names
        self.seed = seed
        self.tree = None

    def fit(self, X, y):
        """ Fits model to provided training dataset.

        Args:
            X: Training samples.
            y: Training labels.
        """

        if is_classifier(self):
            self.tree = DecisionTreeClassifier(ccp_alpha=self.prune_factor, random_state=self.seed)
        else:
            self.tree = DecisionTreeRegressor(ccp_alpha=self.prune_factor, random_state=self.seed)
        self.tree.fit(X, y)
        return self

    def get_model_summary(self):
        return f"PrDT Summary: Leaves number = {self.tree.get_n_leaves()} | Nodes number = {self.tree.tree_.node_count} | Depth = {self.tree.get_depth()}"

    def visualize(self, n_features=None, path=None):
        if not path:
            path = "./prdt_visualized"

        dot_data = export_graphviz(self.tree, out_file=None, feature_names=self.feature_names, filled=True)
        out_graph = pydotplus.graph_from_dot_data(dot_data)
        out_graph.write_svg(f"{path}.svg")

        logger.info(f'Model visualization file is saved at "{path}.svg"')

    def local_visualize(self, X, y, idx=0, n_features=10, path=None):
        logger.info(f"PrDT model not support local visualization, using global visualization instead.")
        self.visualize(n_features=n_features, path=path)


class PrDTClassifier(BasePrDTModel, ClassifierMixin):
    """PrDT Classifier
    """

    def __init__(
        self, *, score_fn="accuracy", penalty_factor=0, prune_factor=0, transformer=None, feature_names=None, seed=123,
    ):
        """PrDT Classifier

        Args:
            score_fn (str): Score function. Support most common metrics likes accuracy, f1, mae, mape. Defaults to "accuracy".
            penalty_factor (float): Penalty factor of model scale. The larger value prefer more compact model. Defaults to 0.
            prune_factor (float): Prune factor for decision tree pruning with ccp. The larger value prefer more compact model. Defaults to 0.
            transformer ([type], optional): Apply for Distill Engine for feature transfer. Defaults to None.
            seed (int, optional): Random state. Defaults to 123.
        """

        super(PrDTClassifier, self).__init__(
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            prune_factor=prune_factor,
            transformer=transformer,
            feature_names=feature_names,
            seed=seed,
        )

    def score(self, X, y):
        f_score = score_mapper(self.score_fn.lower())
        score = f_score(self.predict(X), y) + self.penalty_factor / (
            np.power(self.tree.get_n_leaves() * self.tree.get_depth(), 0.5) + 1e-5
        )
        return score

    def predict(self, X):
        prediction = self.tree.predict(X)
        return prediction


class PrDTRegressor(BasePrDTModel, RegressorMixin):
    """PrDTRegressor
    """

    def __init__(
        self, *, score_fn="mae", penalty_factor=0, prune_factor=0, transformer=None, feature_names=None, seed=123,
    ):
        """PrDT Regressor

        Args:
            score_fn (str): Score function. Support most common metrics likes accuracy, f1, mae, mape. Defaults to "accuracy".
            penalty_factor (float): Penalty factor of model scale. The larger value prefer more compact model. Defaults to 0.
            prune_factor (float): Prune factor for decision tree pruning with ccp. The larger value prefer more compact model. Defaults to 0.
            transformer ([type], optional): Apply for Distill Engine for feature transfer. Defaults to None.
            seed (int, optional): Random state. Defaults to 123.
        """

        super(PrDTRegressor, self).__init__(
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            prune_factor=prune_factor,
            transformer=transformer,
            feature_names=feature_names,
            seed=seed,
        )

    def score(self, X, y):
        f_score = score_mapper(self.score_fn.lower())
        score = f_score(self.predict(X), y) + self.penalty_factor * np.power(
            self.tree.get_n_leaves() * self.tree.get_depth(), 0.5
        )
        return -1 * score

    def predict(self, X):
        prediction = self.tree.predict(X)
        return prediction

import logging
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, state_transformer=None, max_depth=None, max_leaf_nodes=None, prune_factor=None, penalty_factor=0.1):
        self.state_transformer = state_transformer
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.prune_factor = prune_factor
        self.penalty_factor = penalty_factor
        self.tree = None

    def score(self, obss, acts):
        leaves = self.tree.get_n_leaves()
        fidelity = accuracy(self, obss, acts)
        # normalized_mutual_info_score(self.tree.predict(obss), acts)
        score = fidelity + self.penalty_factor / np.sqrt(leaves)
        return score

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes, ccp_alpha=self.prune_factor
        )
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        # For train_dagger without automl
        obss_train, obss_test, acts_train, acts_test = train_test_split(obss, acts, train_size=train_frac, shuffle=True)
        self.fit(obss_train, acts_train)

        logger.info(
            f"Train accuracy: {accuracy(self, obss_train, acts_train):5.3f} | Test accuracy: {accuracy(self, obss_test, acts_test):5.3f} | Leaves number: {self.tree.get_n_leaves()}"
        )

    def predict(self, obss):
        return self.tree.predict(obss)

    def dt_predict_wrapper(self, obss):
        if self.state_transformer is None:
            dt_obss = obss
        else:
            dt_obss = self.state_transformer(obss)
        return self.tree.predict(np.array(dt_obss).reshape(1, -1))[0]

    def clone(self):
        clone = DecisionTree(self.state_transformer, self.max_depth, self.max_leaf_nodes, self.prune_factor)
        clone.tree = self.tree
        return clone


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))

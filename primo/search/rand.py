import logging
from typing import Callable, Dict, Optional, Union, Type

from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)


class RandomSearch:
    def __init__(
        self,
        estimator: Optional[Union[Type, Callable]] = None,
        space: Optional[Dict] = None,
        mode: Optional[str] = None,
        budget: int = 10,
        concurrent_jobs: int = -1,
        random_state: int = 42,
        random_init_points: int = 10,
        verbose: int = 0,
    ):
        assert space is not None, "`space` must be specified for searching"
        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."

        self.estimator = estimator
        self.space = space
        self.verbose = verbose
        self.random_state = random_state
        self.random_init_points = random_init_points
        self.concurrent_jobs = concurrent_jobs
        self.budget = budget

        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0

        cv = [(slice(None), slice(None))]
        self.optimizer = RandomizedSearchCV(
            estimator=self.estimator, param_distributions=self.space, n_jobs=self.concurrent_jobs, cv=cv, refit=True
        )

    def search(self, X, y):
        self.optimizer.fit(X, y)
        return self.optimizer.best_estimator_, self.optimizer.best_params_, self.optimizer.best_score_

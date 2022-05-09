from typing import Callable, Dict, Optional, Union, Type

import logging
from skopt import BayesSearchCV

logger = logging.getLogger(__name__)


class BayesSearch:
    def __init__(
        self,
        estimator: Optional[Union[Type, Callable]] = None,
        space: Optional[Dict] = None,
        budget: int = 10,
        concurrent_jobs: int = -1,
        random_state: int = 123,
        verbose: int = 2,
    ):
        assert space is not None, "`space` must be specified for searching"

        self.estimator = estimator
        self.space = space
        self.random_state = random_state

        self.concurrent_jobs = concurrent_jobs
        self.budget = budget
        self.verbose = verbose

        self.optimizer = BayesSearchCV(
            estimator=self.estimator, search_spaces=self.space, n_iter=self.budget, n_jobs=self.concurrent_jobs, refit=True,
        )

    def search(self, X, y):
        self.optimizer.fit(X, y)
        return self.optimizer.best_estimator_, self.optimizer.best_params_, self.optimizer.best_score_


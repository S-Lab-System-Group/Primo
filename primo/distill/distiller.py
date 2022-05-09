from typing import Any, Callable, Optional, Union, List, Dict

import time
import pickle
import pydotplus
import logging
import numpy as np

from pathlib import Path
from sklearn.tree import export_graphviz
from sklearn.utils.fixes import loguniform
from skopt.space import Real, Categorical, Integer

from primo.utils import logger_init
from primo.distill.dagger import train_dagger, automl_train_dagger
from primo.search import BayesSearch, GridSearch, RandomSearch
from primo.distill.decision_tree import DecisionTree


logger = logging.getLogger(__name__)


def save_decision_tree(decision_tree, dir, file, feature_names):
    with open(f"{dir}/{file}.pkl", "wb") as f:
        pickle.dump(decision_tree.tree, f)

    # Visualizing
    dot_data = export_graphviz(decision_tree.tree, out_file=None, feature_names=feature_names, filled=True)
    out_graph = pydotplus.graph_from_dot_data(dot_data)
    out_graph.write_svg(f"{dir}/{file}.svg")


def load_decision_tree(dir, file):
    with open(f"{dir}/{file}", "rb") as f:
        decision_tree = pickle.load(f)
    return decision_tree


class DistillEngine:
    def __init__(
        self,
        name: Optional[str] = "primo",
        result_path: Optional[str] = "./primo_results",
        rl_test_wrapper: Optional[Callable] = None,
        predict_wrapper: Optional[Callable] = None,
        state_transformer: Optional[Callable] = None,
        feature_names: Optional[str] = None,
        prune_factor: Optional[float] = None,
        search_method: Optional[str] = "auto",
        search_budget: int = 50,
        concurrent_jobs: int = -1,
        seed: int = 1,
        verbose: int = 0,
        framework: str = "tensorflow",
        environment: Any = None,
        actor: Any = None,
        session: Any = None,
    ) -> None:
        # self.env = environment
        # self.actor = actor
        # self.session = session
        self.name = name
        self.result_path = Path(result_path)
        self.feature_names = feature_names
        self.rl_test_wrapper = rl_test_wrapper
        self.state_transformer = state_transformer
        self.predict_wrapper = predict_wrapper
        self.seed = seed

        if not self.result_path.exists():
            self.result_path.mkdir()
        logger_init(self.result_path / self.name)

        # Decision Tree parameters
        if prune_factor is not None:
            logger.info(f"Distill with prune_factor = {prune_factor}, auto parameter search is disabled.")
            search_method = "fix"
            search_budget = 1
        self.prune_factor = prune_factor
        self.search_method = search_method
        self.search_budget = search_budget
        self.concurrent_jobs = concurrent_jobs
        self.verbose = verbose

        if framework != "tensorflow":
            raise NotImplementedError("Only Tensorflow is supportted.")

        self.start_time = time.time()

    def create_seacher(self, **kwargs):
        seacher_list = {"grid": GridSearch, "random": RandomSearch, "bayes": BayesSearch}
        seacher_name = self.search_method.lower()

        # TODO: set search method according to budget number
        if seacher_name == "auto":
            if self.search_budget < 10:
                seacher_name = "grid"
            else:
                seacher_name = "bayes"
        elif seacher_name not in seacher_list:
            raise ValueError(f"Search algorithms should be one of {list(seacher_list)}")

        SearcherClass = seacher_list[seacher_name]

        return SearcherClass(**kwargs)

    def quantize(self, values: Optional[Union[List[Dict], Dict]], quant: float):
        quantized = np.round(np.divide(values, quant)) * quant
        return list(quantized)

    def generate_search_space(self):
        seacher_name = self.search_method.lower()

        if self.prune_factor is not None:
            return
        elif seacher_name == "auto":
            if self.search_budget < 10:
                seacher_name = "grid"
            else:
                seacher_name = "bayes"

        if seacher_name == "grid":
            # space_values = loguniform(1e-6, 1).rvs()
            logmin = np.log10(1e-6)
            logmax = np.log10(1 - 1e-6)

            items = 10 ** (np.random.uniform(logmin, logmax, size=self.search_budget))
            space_values = self.quantize(items, 1e-6)
            return space_values
        elif seacher_name == "random":
            pass
        elif seacher_name == "bayes":
            space_values = Real(1e-8, 1e1, prior="log-uniform")
            return space_values

    def distill(
        self,
        epochs: int = 10,
        train_frac: float = 0.8,
        max_samples: int = 200000,
        n_batch_rollouts: int = 10,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
    ):
        # Step 1: Generate some supervised traces
        trajectories = []
        for _ in range(n_batch_rollouts):
            trajectories.extend(self.rl_test_wrapper(self.predict_wrapper))
        logger.info(f"Collect {len(trajectories)} point trajectories from RL policy.")

        # Step 2: Configure hyperparameter search space
        search_space = self.generate_search_space()

        # Step 3:
        if self.prune_factor is not None:
            best_student = train_dagger(
                trajectories=trajectories,
                state_transformer=self.state_transformer,
                rl_test_wrapper=self.rl_test_wrapper,
                predict_wrapper=self.predict_wrapper,
                prune_factor=self.prune_factor,
                max_depth=max_depth,
                max_leaves=max_leaves,
                epochs=epochs,
                train_frac=train_frac,
                max_samples=max_samples,
                seed=self.seed,
            )
        else:
            assert self.search_method is not "fix"
            logger.info(f"Distill with {self.search_method} search (budget = {self.search_budget}).")

            student = DecisionTree(state_transformer=self.state_transformer, max_depth=max_depth, max_leaf_nodes=max_leaves)

            searcher = self.create_seacher(
                estimator=student,
                space={"prune_factor": search_space},
                mode="max",
                budget=self.search_budget,
                concurrent_jobs=self.concurrent_jobs,
                random_state=self.seed,
                verbose=self.verbose,
            )

            best_student = automl_train_dagger(
                trajectories=trajectories,
                state_transformer=self.state_transformer,
                rl_test_wrapper=self.rl_test_wrapper,
                predict_wrapper=self.predict_wrapper,
                searcher=searcher,
                epochs=epochs,
                train_frac=train_frac,
                max_samples=max_samples,
                seed=self.seed,
            )

        # Step 4:
        save_decision_tree(best_student, self.result_path, self.name, self.feature_names)
        logger.info(f"Distill complete! Total used time: {(time.time() - self.start_time):.2f}(s)")

import logging
import time
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

from skopt.space import Real, Integer
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier

from primo.utils import logger_init, score_mapper
from primo.model import PrAMClassifier, PrAMRegressor, PrDTClassifier, PrDTRegressor
from primo.search import BayesSearch

logger = logging.getLogger(__name__)

PRIMO_MODEL_LIST = ["PRDT", "PRAM"]
CRITERION_LIST = ["PERFORMANCE", "COMPACT"]
SEARCH_LIST = ["BAYES", "RANDOM", "GRID"]

DEFAULT_PRDT_SPACE = {"prune_factor": Real(1e-8, 1e-1, prior="log-uniform")}
DEFAULT_PRAM_SPACE = {
    "max_bins": Integer(2, 256),
    "interactions": Integer(0, 50),
    "learning_rate": Real(1e-4, 1e1, prior="log-uniform"),
}


class BasePrimoModel(BaseEstimator, metaclass=ABCMeta):
    """Base Class for Primo Interpretable Models
    """

    @abstractmethod
    def __init__(
        self,
        *,
        model,
        model_config,
        score_fn,
        penalty_factor,
        feature_names,
        select_criterion,
        hpo,
        hpo_space,
        hpo_budget,
        concurrent_jobs,
        seed,
        verbose,
        logfile,
    ):
        self.model = model
        self.model_config = model_config
        self.score_fn = score_fn
        self.penalty_factor = penalty_factor
        self.feature_names = feature_names
        self.select_criterion = select_criterion
        self.hpo = hpo
        self.hpo_space = hpo_space
        self.hpo_budget = hpo_budget
        self.concurrent_jobs = concurrent_jobs
        self.seed = seed
        self.verbose = verbose
        self.logfile = logfile
        self.prModel = None
        self.prOptimizer = None
        logger_init(self.logfile)

    def model_specify(self, model_type, config=None, classify=True):
        if classify:
            if model_type.upper() == "PRAM":
                logger.info("Using `PrAMClassifier` model for training.")
                if config is None:
                    self.prModel = PrAMClassifier(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                    )
                else:
                    self.prModel = PrAMClassifier(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                        **self.model_config,
                    )
            else:
                logger.info("Using `PrDTClassifier` model for training.")
                if config is None:
                    self.prModel = PrDTClassifier(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                    )
                else:
                    self.prModel = PrDTClassifier(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                        **self.model_config,
                    )
        else:
            if model_type.upper() == "PRAM":
                logger.info("Using `PrAMRegressor` model for training.")
                if config is None:
                    self.prModel = PrAMRegressor(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                    )
                else:
                    self.prModel = PrAMRegressor(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                        **self.model_config,
                    )
            else:
                logger.info("Using `PrDTRegressor` model for training.")
                if config is None:
                    self.prModel = PrDTRegressor(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                    )
                else:
                    self.prModel = PrDTRegressor(
                        score_fn=self.score_fn,
                        penalty_factor=self.penalty_factor,
                        feature_names=self.feature_names,
                        seed=self.seed,
                        **self.model_config,
                    )

    def model_select(self, X, y, Xtest=None, ytest=None):
        logger.info("Start Primo model auto selection.")
        if Xtest is None or ytest is None:
            raise ValueError("`Xtest` and `ytest` value need to specify for model auto selection.")
        if is_classifier(self):
            self.candiAM = PrAMClassifier(
                score_fn=self.score_fn, penalty_factor=self.penalty_factor, feature_names=self.feature_names, seed=self.seed,
            )
            self.candiDT = PrDTClassifier(
                score_fn=self.score_fn, penalty_factor=self.penalty_factor, feature_names=self.feature_names, seed=self.seed,
            )
            mode = 1  # The higher the better
        elif not is_classifier(self):
            self.candiAM = PrAMRegressor(
                score_fn=self.score_fn, penalty_factor=self.penalty_factor, feature_names=self.feature_names, seed=self.seed,
            )
            self.candiDT = PrDTRegressor(
                score_fn=self.score_fn, penalty_factor=self.penalty_factor, feature_names=self.feature_names, seed=self.seed,
            )
            mode = -1  # The lower the better

        f_score = score_mapper(self.score_fn.lower())

        self.candiDT = self.candiDT.fit(X, y)
        start_time = time.perf_counter()
        perfDT = f_score(self.candiDT.predict(Xtest), ytest)
        timeDT = time.perf_counter() - start_time

        self.candiAM = self.candiAM.fit(X, y)
        start_time = time.perf_counter()
        perfAM = f_score(self.candiAM.predict(Xtest), ytest)
        timeAM = time.perf_counter() - start_time

        if self.select_criterion.upper() == "PERFORMANCE":
            if (perfDT * mode) >= (perfAM * mode):
                logger.info(f"PrDT have better performance. | {self.score_fn.upper()} (PrDT: {perfDT}, PrAM: {perfAM})")
                self.model_specify("PrDT", self.model_config, is_classifier(self))
            else:
                logger.info(f"PrDT have better performance. | {self.score_fn.upper()} (PrDT: {perfDT}, PrAM: {perfAM})")
                self.model_specify("PrAM", self.model_config, is_classifier(self))

        elif self.select_criterion.upper() == "COMPACT":
            if timeDT <= timeAM * mode:
                logger.info(f"PrAM have lower inference latency. | TIME (PrDT: {timeDT}, PrAM: {timeAM})")
                self.model_specify("PrDT", self.model_config, is_classifier(self))
            else:
                logger.info(f"PrAM have better performance. | TIME (PrDT: {perfDT}, PrAM: {perfAM})")
                self.model_specify("PrAM", self.model_config, is_classifier(self))
        else:
            raise NotImplementedError

    def fit(self, X, y, Xtest=None, ytest=None):
        """ Fits model to provided training dataset.

        Args:
            X: Training samples.
            y: Training labels.
            Xtest: Test samples for model selection and tuning.
            ytest: Test labels for model selection and tuning.
        """

        # Arguments check
        if self.model is not None:
            assert self.model.upper() in PRIMO_MODEL_LIST, "Choose PrAM or PrDT."
        else:
            assert self.model_config is None, "Need specify model type first."
            assert self.select_criterion.upper() in CRITERION_LIST, "Choose Performance or Compact."

        if self.hpo is not None:
            assert self.hpo.upper() in SEARCH_LIST, "Choose Bayes, Random or Grid."
            assert self.hpo_budget > 0, "Trail number should larger than 0."

        # Specify model type and model config. HPO is off.
        if self.model is not None and self.hpo is None or self.model_config is not None:
            if self.model_config is not None:
                logger.info("Training with given model configuration. Model tuning is disabled.")
            elif self.hpo is None:
                logger.info("Training with default model configuration. Model tuning is disabled.")
            self.model_specify(self.model, self.model_config, is_classifier(self))
            self.prModel.fit(X, y)
            return self

        if self.model is not None and self.hpo_space is None and self.hpo.upper() == "BAYES":
            if self.model.upper() == "PRAM":
                self.hpo_space = DEFAULT_PRAM_SPACE
            else:
                self.hpo_space = DEFAULT_PRDT_SPACE

        # Specify model and perform hyperparameter optimization.
        if self.model is not None:
            self.model_specify(self.model, self.model_config, is_classifier(self))
            self.fit_tune(X, y)
        else:
            # Auto select model first.
            self.model_select(X, y, Xtest, ytest)
            self.fit_tune(X, y)

        return self

    def fit_tune(self, X, y):
        """ Fits model to provided training dataset.

        Args:
            X: Training samples.
            y: Training labels.
            Xtest: Test samples for model selection and tuning.
            ytest: Test labels for model selection and tuning.
        """
        if self.hpo.upper() == "BAYES":
            self.prOptimizer = BayesSearch(
                estimator=self.prModel,
                space=self.hpo_space,
                budget=self.hpo_budget,
                concurrent_jobs=self.concurrent_jobs,
                random_state=self.seed,
                verbose=self.verbose,
            )
            best_model, best_params, best_score = self.prOptimizer.search(X, y)
            logger.info(f"Best Parameter: {best_params}")
            logger.info(f"Best Score: {best_score}")
            logger.info(f"{best_model.get_model_summary()}")
            self.prModel = best_model
            return self
        else:
            raise NotImplementedError

    def visualize(self, n_features=10, path=None):
        # `n_features` only works for PrAM model
        self.prModel.visualize(n_features=n_features, path=path)

    def local_visualize(self, X=None, y=None, idx=0, n_features=10, path=None):
        # Only works for PrAM model
        self.prModel.local_visualize(X, y, idx, n_features=n_features, path=path)

    def predict(self, X):
        prediction = self.prModel.predict(X)
        return prediction


class PrimoClassifier(BasePrimoModel, ClassifierMixin):
    """Primo Interpretable Model Classifier
    """

    def __init__(
        self,
        *,
        model=None,
        model_config=None,
        score_fn="accuracy",
        penalty_factor=0,
        feature_names=None,
        select_criterion="performance",
        hpo="bayes",
        hpo_space=None,
        hpo_budget=10,
        concurrent_jobs=-1,
        seed=123,
        verbose=2,
        logfile=None,
    ):
        """Primo Interpretable Model Classifier

        Args:
            model (String, optional): Specify the interpretable model. Choose "PrAM" or "PrDT". Otherwise Primo select model based on `select_criterion`. Defaults to None.
            model_config (Dict, optional): Specify the model configuration (hyperparameter). Defaults to None.
            score_fn (String, optional): Metric to evalute the model. Defaults to "accuracy".
            feature_names (List, optional): List of feature names. Specify for interpretation. Defaults to None.
            select_criterion (str, optional): User provide system preference for model selection. Choose "performance" or "compact". Defaults to "performance".
            hpo (str, optional): Hyperparameter optimization method. Choose "None", "bayes", "random" or "grid". Defaults to "bayes".
            hpo_space (Dict, optional): Hyperparameter search space. Primo provide basic default space for users. Defaults to None.
            hpo_budget (int, optional): Total hyperparameter search trails limitation. Defaults to 50.
            concurrent_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors. Defaults to -1.
            seed (int, optional): Random state. Defaults to 123.
            verbose (int, optional): Sets verbosity level. The higher, the more messages. Defaults to 2.
            logfile (str, optional): Location of model training logs. None represent without file record. Defaults to None.
        """

        super(PrimoClassifier, self).__init__(
            model=model,
            model_config=model_config,
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            feature_names=feature_names,
            select_criterion=select_criterion,
            hpo=hpo,
            hpo_space=hpo_space,
            hpo_budget=hpo_budget,
            concurrent_jobs=concurrent_jobs,
            seed=seed,
            verbose=verbose,
            logfile=logfile,
        )

    def predict(self, X):
        prediction = self.prModel.predict(X)
        return prediction


class PrimoRegressor(BasePrimoModel, RegressorMixin):
    """Primo Interpretable Model Regressor
    """

    def __init__(
        self,
        *,
        model=None,
        model_config=None,
        score_fn="mae",
        penalty_factor=0,
        feature_names=None,
        select_criterion="performance",
        hpo="bayes",
        hpo_space=None,
        hpo_budget=10,
        concurrent_jobs=-1,
        seed=123,
        verbose=2,
        logfile=None,
    ):
        """Primo Interpretable Model Classifier

        Args:
            model (String, optional): Specify the interpretable model. Choose "PrAM" or "PrDT". Otherwise Primo select model based on `select_criterion`. Defaults to None.
            model_config (Dict, optional): Specify the model configuration (hyperparameter). Defaults to None.
            score_fn (String, optional): Metric to evalute the model. Defaults to "mae".
            feature_names (List, optional): List of feature names. Specify for interpretation. Defaults to None.
            select_criterion (str, optional): User provide system preference for model selection. Choose "performance" or "compact". Defaults to "performance".
            hpo (str, optional): Hyperparameter optimization method. Choose "None", "bayes", "random" or "grid". Defaults to "bayes".
            hpo_space (Dict, optional): Hyperparameter search space. Primo provide basic default space for users. Defaults to None.
            hpo_budget (int, optional): Total hyperparameter search trails limitation. Defaults to 50.
            concurrent_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors. Defaults to -1.
            seed (int, optional): Random state. Defaults to 123.
            verbose (int, optional): Sets verbosity level. The higher, the more messages. Defaults to 2.
            logfile (str, optional): Location of model training logs. None represent without file record. Defaults to None.
        """

        super(PrimoRegressor, self).__init__(
            model=model,
            model_config=model_config,
            score_fn=score_fn,
            penalty_factor=penalty_factor,
            feature_names=feature_names,
            select_criterion=select_criterion,
            hpo=hpo,
            hpo_space=hpo_space,
            hpo_budget=hpo_budget,
            concurrent_jobs=concurrent_jobs,
            seed=seed,
            verbose=verbose,
            logfile=logfile,
        )

    def predict(self, X):
        prediction = self.prModel.predict(X)
        return prediction

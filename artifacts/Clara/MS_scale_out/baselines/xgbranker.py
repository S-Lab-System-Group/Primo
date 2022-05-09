import xgboost
from xgboost import XGBModel
from xgboost import DMatrix, train
import numpy as np


class XGBRanker(XGBModel):
    __doc__ = """Implementation of sklearn API for XGBoost Ranking
           """ + "\n".join(
        XGBModel.__doc__.split("\n")[2:]
    )

    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        objective="rank:pairwise",
        booster="gbtree",
        n_jobs=-1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None,
        missing=None,
        **kwargs
    ):

        super(XGBRanker, self).__init__(
            max_depth,
            learning_rate,
            n_estimators,
            silent,
            objective,
            booster,
            n_jobs,
            nthread,
            gamma,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            reg_alpha,
            reg_lambda,
            scale_pos_weight,
            base_score,
            random_state,
            seed,
            missing,
        )

    def fit(self, X, y, group=None, eval_metric=None, sample_weight=None, early_stopping_rounds=None, verbose=True):
        if group is None:
            group = [X.shape[0]]
        else:
            idx = np.argsort(group)
            X = X[idx, :]
            y = y[idx]
            group = group[idx]
            unique, counts = np.unique(group, return_counts=True)
            group = counts[np.argsort(unique)]

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            xgb_options["objective"] = "rank:pairwise"
        else:
            obj = None

        evals_result = {}
        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({"eval_metric": eval_metric})

        if sample_weight is not None:
            train_dmatrix = DMatrix(X, label=y, weight=sample_weight, missing=self.missing)
        else:
            train_dmatrix = DMatrix(X, label=y, missing=self.missing)
        train_dmatrix.set_group(group)

        self.objective = params["objective"]

        self._Booster = train(
            params,
            train_dmatrix,
            self.n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=None,
        )

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    def predict(self, X, group=None, output_margin=False, ntree_limit=0):
        unsort = group is not None
        idx = np.argsort(group)
        X = X[idx, :]
        # print group, idx
        group = group[idx]
        unique, counts = np.unique(group, return_counts=True)
        group = counts[np.argsort(unique)]

        test_dmatrix = DMatrix(X, missing=self.missing)
        test_dmatrix.set_group(group)
        rank_values = self.get_booster().predict(test_dmatrix, output_margin=output_margin, ntree_limit=ntree_limit)
        if unsort:
            rank_values = rank_values[np.argsort(idx)]
        return rank_values

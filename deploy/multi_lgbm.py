from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class MultiLGBMRegressor:
    """Multi-output LightGBM with early stopping (copied from main package)."""

    def __init__(
        self,
        *,
        base_params: dict | None = None,
        upper_bound_estimators: int = 2000,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.base_params = base_params or {}
        self.upper_bound_estimators = upper_bound_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.estimators_: List[LGBMRegressor] = []
        self.best_iterations_: List[int] = []
        self.best_iteration_: int = 0
        self.feature_names_in_: List[str] = []

    # ---------------------------------------------------
    # scikit-learn compatible API
    # ---------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        Y: pd.DataFrame | np.ndarray,
        *,
        eval_set: Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray] | None = None,
        eval_metric: str = "l1",
        verbose: bool | int = False,
    ):
        def _col(arr, idx):
            if hasattr(arr, "iloc"):
                return arr.iloc[:, idx]
            return arr[:, idx]

        n_outputs = Y.shape[1] if getattr(Y, "ndim", 1) > 1 else 1
        self.estimators_.clear()
        self.best_iterations_.clear()

        for idx in range(n_outputs):
            y_col = _col(Y, idx) if n_outputs > 1 else Y

            params = {
                "n_estimators": self.upper_bound_estimators,
                "random_state": 42 + idx,
                "n_jobs": -1,
            }
            params.update(self.base_params)
            mdl = LGBMRegressor(**params)

            if eval_set is not None:
                X_val, Y_val = eval_set
                y_val = _col(Y_val, idx) if n_outputs > 1 else Y_val
                mdl.fit(
                    X,
                    y_col,
                    eval_set=[(X_val, y_val)],
                    eval_metric=eval_metric,
                    callbacks=[
                        __import__("lightgbm").early_stopping(
                            self.early_stopping_rounds,
                            first_metric_only=True,
                            verbose=bool(verbose),
                        )
                    ],
                )
            else:
                mdl.fit(X, y_col)

            self.estimators_.append(mdl)
            bi = getattr(mdl, "best_iteration_", None) or mdl.n_estimators_
            self.best_iterations_.append(int(bi))
            if idx == 0 and hasattr(mdl, "feature_name_"):
                self.feature_names_in_ = list(mdl.feature_name_)

        self.best_iteration_ = int(np.mean(self.best_iterations_))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray):
        preds = [est.predict(X) for est in self.estimators_]
        return np.column_stack(preds) if len(preds) > 1 else preds[0]

    @property
    def feature_importances_(self):
        if not self.estimators_:
            return np.array([])
        imps = np.vstack([est.feature_importances_ for est in self.estimators_])
        return imps.mean(axis=0) 
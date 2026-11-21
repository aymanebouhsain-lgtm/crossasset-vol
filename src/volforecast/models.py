from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

FEATURES = ["rv_d", "rv_w", "rv_m"]


class BaseModel(ABC):
    """Common interface: every model learns in `fit` and forecasts in `predict`."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Learn from training features X and target y. Returns self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Forecast the target for each row of X (same index as X)."""


class HARModel(BaseModel):
    """HAR-RV: y = beta_0 + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m, fitted by OLS."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        Z = np.column_stack([np.ones(len(X)), X[FEATURES].to_numpy()])
        beta = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
        self.coef_ = pd.Series(beta, index=["beta_0", "beta_d", "beta_w", "beta_m"])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        A = np.column_stack([np.ones(len(X)), X[FEATURES].to_numpy()])
        y_hat = A @ self.coef_.to_numpy()
        return pd.Series(y_hat, index=X.index, name="forecast")


class NaiveModel(BaseModel):
    """Next `horizon` days' variance = the same as the last `horizon` days (horizon * rv_w)."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveModel":
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.horizon * X["rv_w"]).rename("forecast")


class MeanModel(BaseModel):
    """Predicts the training-set average of the target for every date."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanModel":
        self.mean_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.mean_, index=X.index, name="forecast")
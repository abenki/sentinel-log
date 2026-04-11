"""Isolation Forest baseline model for anomaly detection."""

import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib


class IsolationForestBaseline:
    """Isolation Forest baseline model for anomaly detection."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """Initializes the Isolation Forest model with the given parameters."""
        self.model = IsolationForest(
            contamination=contamination,  # type: ignore[arg-type]
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.threshold_: float | None = None
        self.train_score_min_: float | None = None
        self.train_score_max_: float | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fits the model on normal-only data and computes the anomaly threshold.

        Stores the min and max of the training scores for min-max normalization,
        and sets the threshold at the 95th percentile of normalized anomaly scores.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Should contain
                normal samples only to avoid contaminating the baseline.
        """
        self.model.fit(X)
        anomaly_scores = self.model.score_samples(X)
        self.train_score_min_ = anomaly_scores.min()
        self.train_score_max_ = anomaly_scores.max()
        normalized_scores = self.anomaly_score(X)
        self.threshold_ = np.percentile(normalized_scores, 95)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores in [0, 1] via min-max scaling on the training set.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores in [0, 1] for each sample.
        """
        assert (
            self.train_score_min_ is not None and self.train_score_max_ is not None
        ), "Model must be fitted before calling anomaly_score()"

        if self.train_score_min_ == self.train_score_max_:
            return np.full(X.shape[0], 0.5)

        anomaly_scores = self.model.score_samples(X)
        rescaled = (anomaly_scores - self.train_score_min_) / (
            self.train_score_max_ - self.train_score_min_
        )
        return 1.0 - rescaled

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Returns boolean array: score > threshold_.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Boolean array indicating whether each sample is anomalous.
        """
        assert self.threshold_ is not None, (
            "Model must be fitted before calling predict_labels()"
        )
        return self.anomaly_score(X) > self.threshold_

    def save(self, path: Path) -> None:
        """Saves the model to the given path using joblib.

        Args:
            path: Path to save the model to.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestBaseline":
        """Loads the model from the given path using joblib.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model instance.
        """
        return joblib.load(path)

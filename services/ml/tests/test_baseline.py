"""Tests for the Isolation Forest baseline model."""

import numpy as np
import pytest
from pathlib import Path

from baseline import IsolationForestBaseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_data() -> np.ndarray:
    """Returns a normal-only feature matrix for training."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(500, 3))


@pytest.fixture
def anomaly_data() -> np.ndarray:
    """Returns an obvious anomaly feature matrix (far from normal distribution)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=10.0, scale=0.1, size=(50, 3))


@pytest.fixture
def fitted_model(normal_data: np.ndarray) -> IsolationForestBaseline:
    """Returns an IsolationForestBaseline fitted on normal data."""
    model = IsolationForestBaseline(
        contamination=0.05, n_estimators=100, random_state=42
    )
    model.fit(normal_data)
    return model


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_runs_without_error(normal_data: np.ndarray) -> None:
    """fit() completes without raising on normal data."""
    model = IsolationForestBaseline(random_state=42)
    model.fit(normal_data)


def test_fit_sets_threshold(fitted_model: IsolationForestBaseline) -> None:
    """fit() sets threshold_ to a non-None float value."""
    assert fitted_model.threshold_ is not None
    assert isinstance(fitted_model.threshold_, float)


def test_fit_sets_train_score_min(fitted_model: IsolationForestBaseline) -> None:
    """fit() sets train_score_min_ to a non-None float value."""
    assert fitted_model.train_score_min_ is not None
    assert isinstance(fitted_model.train_score_min_, float)


def test_fit_sets_train_score_max(fitted_model: IsolationForestBaseline) -> None:
    """fit() sets train_score_max_ to a non-None float value."""
    assert fitted_model.train_score_max_ is not None
    assert isinstance(fitted_model.train_score_max_, float)


def test_fit_min_less_than_max(fitted_model: IsolationForestBaseline) -> None:
    """fit() sets train_score_min_ strictly less than train_score_max_."""
    assert fitted_model.train_score_min_ < fitted_model.train_score_max_  # type: ignore[operator]


# ---------------------------------------------------------------------------
# anomaly_score()
# ---------------------------------------------------------------------------


def test_anomaly_score_shape(
    fitted_model: IsolationForestBaseline, normal_data: np.ndarray
) -> None:
    """anomaly_score() returns an array of shape (n_samples,)."""
    scores = fitted_model.anomaly_score(normal_data)
    assert scores.shape == (normal_data.shape[0],)


def test_anomaly_score_range(
    fitted_model: IsolationForestBaseline, normal_data: np.ndarray
) -> None:
    """anomaly_score() returns values in [0, 1]."""
    scores = fitted_model.anomaly_score(normal_data)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_anomaly_score_anomalies_higher_than_normal(
    fitted_model: IsolationForestBaseline,
    normal_data: np.ndarray,
    anomaly_data: np.ndarray,
) -> None:
    """anomaly_score() assigns higher mean scores to anomalies than to normal samples."""
    normal_scores = fitted_model.anomaly_score(normal_data)
    anomaly_scores = fitted_model.anomaly_score(anomaly_data)
    assert anomaly_scores.mean() > normal_scores.mean()


def test_anomaly_score_raises_before_fit() -> None:
    """anomaly_score() raises AssertionError if called before fit()."""
    model = IsolationForestBaseline()
    X = np.zeros((10, 3))
    with pytest.raises(AssertionError, match="Model must be fitted"):
        model.anomaly_score(X)


def test_anomaly_score_constant_returns_half(
    fitted_model: IsolationForestBaseline,
) -> None:
    """anomaly_score() returns 0.5 for all samples when min equals max."""
    fitted_model.train_score_min_ = 0.5
    fitted_model.train_score_max_ = 0.5
    X = np.zeros((5, 3))
    scores = fitted_model.anomaly_score(X)
    assert np.all(scores == 0.5)


# ---------------------------------------------------------------------------
# predict_labels()
# ---------------------------------------------------------------------------


def test_predict_labels_returns_bool_array(
    fitted_model: IsolationForestBaseline, normal_data: np.ndarray
) -> None:
    """predict_labels() returns a boolean numpy array."""
    labels = fitted_model.predict_labels(normal_data)
    assert labels.dtype == bool


def test_predict_labels_shape(
    fitted_model: IsolationForestBaseline, normal_data: np.ndarray
) -> None:
    """predict_labels() returns an array of shape (n_samples,)."""
    labels = fitted_model.predict_labels(normal_data)
    assert labels.shape == (normal_data.shape[0],)


def test_predict_labels_raises_before_fit() -> None:
    """predict_labels() raises AssertionError if called before fit()."""
    model = IsolationForestBaseline()
    X = np.zeros((10, 3))
    with pytest.raises(AssertionError, match="Model must be fitted"):
        model.predict_labels(X)


def test_predict_labels_detects_obvious_anomalies(
    fitted_model: IsolationForestBaseline,
    anomaly_data: np.ndarray,
) -> None:
    """predict_labels() flags the majority of obvious anomalies as anomalous."""
    labels = fitted_model.predict_labels(anomaly_data)
    assert labels.mean() > 0.5


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


def test_save_creates_file(
    fitted_model: IsolationForestBaseline, tmp_path: Path
) -> None:
    """save() creates a file at the specified path."""
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    assert path.exists()


def test_load_returns_instance(
    fitted_model: IsolationForestBaseline, tmp_path: Path
) -> None:
    """load() returns an IsolationForestBaseline instance."""
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    loaded = IsolationForestBaseline.load(path)
    assert isinstance(loaded, IsolationForestBaseline)


def test_save_load_roundtrip_scores(
    fitted_model: IsolationForestBaseline,
    normal_data: np.ndarray,
    tmp_path: Path,
) -> None:
    """save/load round trip produces identical anomaly scores."""
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    loaded = IsolationForestBaseline.load(path)
    original_scores = fitted_model.anomaly_score(normal_data)
    loaded_scores = loaded.anomaly_score(normal_data)
    np.testing.assert_array_equal(original_scores, loaded_scores)


def test_save_load_roundtrip_threshold(
    fitted_model: IsolationForestBaseline, tmp_path: Path
) -> None:
    """save/load round trip preserves the threshold value."""
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    loaded = IsolationForestBaseline.load(path)
    assert loaded.threshold_ == fitted_model.threshold_

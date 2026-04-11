"""Tests for the feature extraction module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from features import extract_features
from shared.schemas import DbLog, HttpLog, LogEntry, LogType, SystemLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def http_log_entry() -> LogEntry:
    """Returns a valid HTTP log entry."""
    return LogEntry(
        timestamp=datetime.now(tz=timezone.utc),
        log_type=LogType.HTTP,
        source="web-01",
        payload=HttpLog(
            type="http",
            method="GET",
            path="/api/users",
            status_code=200,
            latency_ms=150.0,
            user_agent="curl/7.68.0",
        ),
        is_anomaly=False,
    )


@pytest.fixture
def system_log_entry() -> LogEntry:
    """Returns a valid system log entry."""
    return LogEntry(
        timestamp=datetime.now(tz=timezone.utc),
        log_type=LogType.SYSTEM,
        source="worker-01",
        payload=SystemLog(
            type="system",
            cpu_pct=45.0,
            mem_pct=60.0,
            disk_io_mb=50.0,
            hostname="worker-01",
        ),
        is_anomaly=False,
    )


@pytest.fixture
def db_log_entry() -> LogEntry:
    """Returns a valid DB log entry."""
    return LogEntry(
        timestamp=datetime.now(tz=timezone.utc),
        log_type=LogType.DB,
        source="postgres-primary",
        payload=DbLog(
            type="db",
            query="SELECT * FROM users WHERE id = ?",
            duration_ms=25.0,
            rows_returned=10,
            db_name="postgres-primary",
        ),
        is_anomaly=False,
    )


# ---------------------------------------------------------------------------
# HTTP features
# ---------------------------------------------------------------------------


def test_extract_features_http_keys(http_log_entry: LogEntry) -> None:
    """extract_features returns exactly the expected keys for HTTP log entries."""
    features = extract_features(http_log_entry)
    assert set(features.keys()) == {"status_code", "latency_ms"}


def test_extract_features_http_values(http_log_entry: LogEntry) -> None:
    """extract_features returns the correct values for HTTP log entries."""
    features = extract_features(http_log_entry)
    assert features["status_code"] == 200.0
    assert features["latency_ms"] == 150.0


# ---------------------------------------------------------------------------
# SYSTEM features
# ---------------------------------------------------------------------------


def test_extract_features_system_keys(system_log_entry: LogEntry) -> None:
    """extract_features returns exactly the expected keys for system log entries."""
    features = extract_features(system_log_entry)
    assert set(features.keys()) == {"cpu_pct", "mem_pct", "disk_io_mb"}


def test_extract_features_system_values(system_log_entry: LogEntry) -> None:
    """extract_features returns the correct values for system log entries."""
    features = extract_features(system_log_entry)
    assert features["cpu_pct"] == 45.0
    assert features["mem_pct"] == 60.0
    assert features["disk_io_mb"] == 50.0


# ---------------------------------------------------------------------------
# DB features
# ---------------------------------------------------------------------------


def test_extract_features_db_keys(db_log_entry: LogEntry) -> None:
    """extract_features returns exactly the expected keys for DB log entries."""
    features = extract_features(db_log_entry)
    assert set(features.keys()) == {"duration_ms", "rows_returned"}


def test_extract_features_db_values(db_log_entry: LogEntry) -> None:
    """extract_features returns the correct values for DB log entries."""
    features = extract_features(db_log_entry)
    assert features["duration_ms"] == 25.0
    assert features["rows_returned"] == 10.0


# ---------------------------------------------------------------------------
# Float casting
# ---------------------------------------------------------------------------


def test_extract_features_http_all_floats(http_log_entry: LogEntry) -> None:
    """All values returned for HTTP log entries are floats."""
    features = extract_features(http_log_entry)
    for key, value in features.items():
        assert isinstance(value, float), f"{key} is not a float"


def test_extract_features_system_all_floats(system_log_entry: LogEntry) -> None:
    """All values returned for system log entries are floats."""
    features = extract_features(system_log_entry)
    for key, value in features.items():
        assert isinstance(value, float), f"{key} is not a float"


def test_extract_features_db_all_floats(db_log_entry: LogEntry) -> None:
    """All values returned for DB log entries are floats."""
    features = extract_features(db_log_entry)
    for key, value in features.items():
        assert isinstance(value, float), f"{key} is not a float"


def test_extract_features_status_code_cast_to_float(http_log_entry: LogEntry) -> None:
    """status_code is cast to float even though it is an int in the schema."""
    features = extract_features(http_log_entry)
    assert type(features["status_code"]) is float


def test_extract_features_rows_returned_cast_to_float(db_log_entry: LogEntry) -> None:
    """rows_returned is cast to float even though it is an int in the schema."""
    features = extract_features(db_log_entry)
    assert type(features["rows_returned"]) is float


# ---------------------------------------------------------------------------
# Unsupported log type
# ---------------------------------------------------------------------------


def test_extract_features_unsupported_log_type_raises() -> None:
    """extract_features raises ValueError for unsupported log types."""
    log = MagicMock()
    log.log_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported log type"):
        extract_features(log)

"""Tests for the Kafka log consumer."""

import json
import pytest
import httpx
import respx

from datetime import datetime, timezone
from unittest.mock import MagicMock

from consumer import LogConsumer
from shared.schemas import LogEntry, LogType, HttpLog, SystemLog, DbLog


ML_SERVICE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def consumer() -> LogConsumer:
    kafka_consumer = MagicMock()
    return LogConsumer(consumer=kafka_consumer, ml_service_url=ML_SERVICE_URL)


@pytest.fixture
def http_log_entry() -> LogEntry:
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
# LogConsumer._extract_features
# ---------------------------------------------------------------------------


def test_extract_features_http(consumer: LogConsumer, http_log_entry: LogEntry) -> None:
    """_extract_features returns all HttpLog fields for an HTTP log entry."""
    features = consumer._extract_features(http_log_entry)
    assert features["method"] == "GET"
    assert features["path"] == "/api/users"
    assert features["status_code"] == 200
    assert features["latency_ms"] == 150.0
    assert features["user_agent"] == "curl/7.68.0"


def test_extract_features_system(
    consumer: LogConsumer, system_log_entry: LogEntry
) -> None:
    """_extract_features returns all SystemLog fields for a system log entry."""
    features = consumer._extract_features(system_log_entry)
    assert features["cpu_pct"] == 45.0
    assert features["mem_pct"] == 60.0
    assert features["disk_io_mb"] == 50.0
    assert features["hostname"] == "worker-01"


def test_extract_features_db(consumer: LogConsumer, db_log_entry: LogEntry) -> None:
    """_extract_features returns all DbLog fields for a DB log entry."""
    features = consumer._extract_features(db_log_entry)
    assert features["query"] == "SELECT * FROM users WHERE id = ?"
    assert features["duration_ms"] == 25.0
    assert features["rows_returned"] == 10
    assert features["db_name"] == "postgres-primary"


def test_extract_features_http_keys(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_extract_features returns exactly the expected keys for HTTP logs."""
    features = consumer._extract_features(http_log_entry)
    assert set(features.keys()) == {
        "method",
        "path",
        "status_code",
        "latency_ms",
        "user_agent",
    }


def test_extract_features_system_keys(
    consumer: LogConsumer, system_log_entry: LogEntry
) -> None:
    """_extract_features returns exactly the expected keys for system logs."""
    features = consumer._extract_features(system_log_entry)
    assert set(features.keys()) == {"cpu_pct", "mem_pct", "disk_io_mb", "hostname"}


def test_extract_features_db_keys(
    consumer: LogConsumer, db_log_entry: LogEntry
) -> None:
    """_extract_features returns exactly the expected keys for DB logs."""
    features = consumer._extract_features(db_log_entry)
    assert set(features.keys()) == {"query", "duration_ms", "rows_returned", "db_name"}


# ---------------------------------------------------------------------------
# LogConsumer._process_message — malformed input
# ---------------------------------------------------------------------------


def test_process_message_invalid_json_skips(consumer: LogConsumer) -> None:
    """_process_message skips and logs on invalid JSON without raising."""
    consumer._process_message(b"not valid json {{{")


def test_process_message_invalid_schema_skips(consumer: LogConsumer) -> None:
    """_process_message skips and logs on Pydantic validation failure without raising."""
    payload = json.dumps({"totally": "wrong", "schema": True}).encode("utf-8")
    consumer._process_message(payload)


# ---------------------------------------------------------------------------
# LogConsumer._process_message — ML service interaction
# ---------------------------------------------------------------------------


@respx.mock
def test_process_message_calls_predict(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_process_message calls POST /predict with the correct payload structure."""
    route = respx.post(f"{ML_SERVICE_URL}/predict").mock(
        return_value=httpx.Response(200, json={"anomaly_score": 0.1})
    )
    raw = http_log_entry.model_dump_json().encode("utf-8")
    consumer._process_message(raw)
    assert route.called


@respx.mock
def test_process_message_payload_structure(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_process_message sends log_type, source, timestamp, and features to /predict."""
    route = respx.post(f"{ML_SERVICE_URL}/predict").mock(
        return_value=httpx.Response(200, json={"anomaly_score": 0.1})
    )
    raw = http_log_entry.model_dump_json().encode("utf-8")
    consumer._process_message(raw)
    sent_body = json.loads(route.calls.last.request.content)
    assert "log_type" in sent_body
    assert "source" in sent_body
    assert "timestamp" in sent_body
    assert "features" in sent_body
    assert isinstance(sent_body["features"], dict)


@respx.mock
def test_process_message_ml_service_down_skips(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_process_message skips without raising when the ML service is unreachable."""
    respx.post(f"{ML_SERVICE_URL}/predict").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    raw = http_log_entry.model_dump_json().encode("utf-8")
    consumer._process_message(raw)


@respx.mock
def test_process_message_ml_service_500_skips(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_process_message skips without raising when the ML service returns 5xx."""
    respx.post(f"{ML_SERVICE_URL}/predict").mock(return_value=httpx.Response(500))
    raw = http_log_entry.model_dump_json().encode("utf-8")
    consumer._process_message(raw)


@respx.mock
def test_process_message_correct_log_type_value(
    consumer: LogConsumer, http_log_entry: LogEntry
) -> None:
    """_process_message sends the string value of log_type, not the enum."""
    route = respx.post(f"{ML_SERVICE_URL}/predict").mock(
        return_value=httpx.Response(200, json={"anomaly_score": 0.1})
    )
    raw = http_log_entry.model_dump_json().encode("utf-8")
    consumer._process_message(raw)
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["log_type"] == "http"

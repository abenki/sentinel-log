"""Tests for the log generator and anomaly injector."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from generator import AnomalyInjector, LogGenerator
from schemas import LogEntry, LogType, HttpLog, SystemLog, DbLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def injector_no_anomaly() -> AnomalyInjector:
    return AnomalyInjector(rate=0.0)


@pytest.fixture
def injector_always_anomaly() -> AnomalyInjector:
    return AnomalyInjector(rate=1.0)


@pytest.fixture
def generator(injector_no_anomaly: AnomalyInjector) -> LogGenerator:
    producer = MagicMock()
    return LogGenerator(injector=injector_no_anomaly, producer=producer)


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
def db_log_entry_select() -> LogEntry:
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


@pytest.fixture
def db_log_entry_insert() -> LogEntry:
    return LogEntry(
        timestamp=datetime.now(tz=timezone.utc),
        log_type=LogType.DB,
        source="postgres-primary",
        payload=DbLog(
            type="db",
            query="INSERT INTO orders (user_id, product_id, quantity) VALUES (?, ?, ?)",
            duration_ms=25.0,
            rows_returned=1,
            db_name="postgres-primary",
        ),
        is_anomaly=False,
    )


# ---------------------------------------------------------------------------
# AnomalyInjector.inject
# ---------------------------------------------------------------------------


def test_inject_rate_zero_never_injects(
    injector_no_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """inject with rate=0.0 always returns the log unchanged."""
    for _ in range(100):
        result = injector_no_anomaly.inject(http_log_entry)
        assert result.is_anomaly is False


def test_inject_rate_one_always_injects(
    injector_always_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """inject with rate=1.0 always returns a log with is_anomaly=True."""
    for _ in range(100):
        result = injector_always_anomaly.inject(http_log_entry)
        assert result.is_anomaly is True


def test_inject_returns_new_instance(
    injector_always_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """inject returns a new LogEntry instance, not a mutation of the original."""
    result = injector_always_anomaly.inject(http_log_entry)
    assert result is not http_log_entry


# ---------------------------------------------------------------------------
# AnomalyInjector._inject_http
# ---------------------------------------------------------------------------


def test_inject_http_sets_is_anomaly(
    injector_always_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """_inject_http returns a log entry with is_anomaly set to True."""
    result = injector_always_anomaly._inject_http(http_log_entry)
    assert result.is_anomaly is True


def test_inject_http_latency_spike(
    injector_always_anomaly: AnomalyInjector,
    http_log_entry: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_http latency spike multiplies latency_ms by 10."""
    monkeypatch.setattr("generator.random.choice", lambda _: "latency_spike")
    assert isinstance(http_log_entry.payload, HttpLog)
    original_latency = http_log_entry.payload.latency_ms
    result = injector_always_anomaly._inject_http(http_log_entry)
    assert isinstance(result.payload, HttpLog)
    assert result.payload.latency_ms == pytest.approx(original_latency * 10)


def test_inject_http_5xx(
    injector_always_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """_inject_http 5xx forces status_code to 500, 502, or 503."""
    with patch("generator.random.choice", side_effect=["5xx", 503]):
        result = injector_always_anomaly._inject_http(http_log_entry)
        assert isinstance(result.payload, HttpLog)
        assert result.payload.status_code in [500, 502, 503]


def test_inject_http_preserves_payload_type(
    injector_always_anomaly: AnomalyInjector, http_log_entry: LogEntry
) -> None:
    """_inject_http always returns an HttpLog payload."""
    result = injector_always_anomaly._inject_http(http_log_entry)
    assert isinstance(result.payload, HttpLog)


# ---------------------------------------------------------------------------
# AnomalyInjector._inject_system
# ---------------------------------------------------------------------------


def test_inject_system_sets_is_anomaly(
    injector_always_anomaly: AnomalyInjector, system_log_entry: LogEntry
) -> None:
    """_inject_system returns a log entry with is_anomaly set to True."""
    result = injector_always_anomaly._inject_system(system_log_entry)
    assert result.is_anomaly is True


def test_inject_system_cpu_spike(
    injector_always_anomaly: AnomalyInjector,
    system_log_entry: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_system cpu_spike sets cpu_pct between 92 and 99."""
    monkeypatch.setattr("generator.random.choice", lambda _: "cpu_spike")
    result = injector_always_anomaly._inject_system(system_log_entry)
    assert isinstance(result.payload, SystemLog)
    assert 92.0 <= result.payload.cpu_pct <= 99.0


def test_inject_system_memory_pressure(
    injector_always_anomaly: AnomalyInjector,
    system_log_entry: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_system memory_pressure sets mem_pct between 90 and 98."""
    monkeypatch.setattr("generator.random.choice", lambda _: "memory_pressure")
    result = injector_always_anomaly._inject_system(system_log_entry)
    assert isinstance(result.payload, SystemLog)
    assert 90.0 <= result.payload.mem_pct <= 98.0


def test_inject_system_preserves_payload_type(
    injector_always_anomaly: AnomalyInjector, system_log_entry: LogEntry
) -> None:
    """_inject_system always returns a SystemLog payload."""
    result = injector_always_anomaly._inject_system(system_log_entry)
    assert isinstance(result.payload, SystemLog)


# ---------------------------------------------------------------------------
# AnomalyInjector._inject_db
# ---------------------------------------------------------------------------


def test_inject_db_sets_is_anomaly(
    injector_always_anomaly: AnomalyInjector, db_log_entry_select: LogEntry
) -> None:
    """_inject_db returns a log entry with is_anomaly set to True."""
    result = injector_always_anomaly._inject_db(db_log_entry_select)
    assert result.is_anomaly is True


def test_inject_db_slow_query(
    injector_always_anomaly: AnomalyInjector,
    db_log_entry_select: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_db slow_query multiplies duration_ms by 20."""
    monkeypatch.setattr("generator.random.choice", lambda _: "slow_query")
    assert isinstance(db_log_entry_select.payload, DbLog)
    original_duration = db_log_entry_select.payload.duration_ms
    result = injector_always_anomaly._inject_db(db_log_entry_select)
    assert isinstance(result.payload, DbLog)
    assert result.payload.duration_ms == pytest.approx(original_duration * 20)


def test_inject_db_zero_rows_on_select(
    injector_always_anomaly: AnomalyInjector,
    db_log_entry_select: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_db zero_rows sets rows_returned to 0 on SELECT queries."""
    monkeypatch.setattr("generator.random.choice", lambda _: "zero_rows")
    result = injector_always_anomaly._inject_db(db_log_entry_select)
    assert isinstance(result.payload, DbLog)
    assert result.payload.rows_returned == 0


def test_inject_db_zero_rows_fallback_on_non_select(
    injector_always_anomaly: AnomalyInjector,
    db_log_entry_insert: LogEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_inject_db zero_rows falls back to slow_query on non-SELECT queries."""
    monkeypatch.setattr("generator.random.choice", lambda _: "zero_rows")
    assert isinstance(db_log_entry_insert.payload, DbLog)
    original_duration = db_log_entry_insert.payload.duration_ms
    result = injector_always_anomaly._inject_db(db_log_entry_insert)
    assert isinstance(result.payload, DbLog)
    assert result.payload.duration_ms == pytest.approx(original_duration * 20)
    assert result.payload.rows_returned != 0


def test_inject_db_preserves_payload_type(
    injector_always_anomaly: AnomalyInjector, db_log_entry_select: LogEntry
) -> None:
    """_inject_db always returns a DbLog payload."""
    result = injector_always_anomaly._inject_db(db_log_entry_select)
    assert isinstance(result.payload, DbLog)


# ---------------------------------------------------------------------------
# LogGenerator._generate_*
# ---------------------------------------------------------------------------


def test_generate_http_returns_http_log(generator: LogGenerator) -> None:
    """_generate_http returns a LogEntry with an HttpLog payload."""
    result = generator._generate_http()
    assert result.log_type == LogType.HTTP
    assert isinstance(result.payload, HttpLog)
    assert result.is_anomaly is False


def test_generate_system_returns_system_log(generator: LogGenerator) -> None:
    """_generate_system returns a LogEntry with a SystemLog payload."""
    result = generator._generate_system()
    assert result.log_type == LogType.SYSTEM
    assert isinstance(result.payload, SystemLog)
    assert result.is_anomaly is False


def test_generate_db_returns_db_log(generator: LogGenerator) -> None:
    """_generate_db returns a LogEntry with a DbLog payload."""
    result = generator._generate_db()
    assert result.log_type == LogType.DB
    assert isinstance(result.payload, DbLog)
    assert result.is_anomaly is False


def test_generate_http_source_is_valid(generator: LogGenerator) -> None:
    """_generate_http picks a source from HTTP_SOURCES."""
    from generator import HTTP_SOURCES

    result = generator._generate_http()
    assert result.source in HTTP_SOURCES


def test_generate_system_hostname_matches_source(generator: LogGenerator) -> None:
    """_generate_system sets hostname equal to source."""
    result = generator._generate_system()
    assert isinstance(result.payload, SystemLog)
    assert result.payload.hostname == result.source


def test_generate_db_db_name_matches_source(generator: LogGenerator) -> None:
    """_generate_db sets db_name equal to source."""
    result = generator._generate_db()
    assert isinstance(result.payload, DbLog)
    assert result.payload.db_name == result.source


def test_generate_http_timestamp_is_utc(generator: LogGenerator) -> None:
    """_generate_http produces a timestamp with UTC timezone."""
    result = generator._generate_http()
    assert result.timestamp.tzinfo is not None
    assert result.timestamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

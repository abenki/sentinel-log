"""Feature extraction for the ML pipeline."""

from shared.schemas import DbLog, HttpLog, LogEntry, LogType, SystemLog


def extract_features(log: LogEntry) -> dict[str, float]:
    """Extract the numeric features vector of a log entry.

    Returns dict with the keys:
    - HTTP  : {"status_code", "latency_ms"}
    - SYSTEM: {"cpu_pct", "mem_pct", "disk_io_mb"}
    - DB    : {"duration_ms", "rows_returned"}
    """
    features = {}
    if log.log_type == LogType.HTTP:
        assert isinstance(log.payload, HttpLog)
        features["status_code"] = float(log.payload.status_code)
        features["latency_ms"] = float(log.payload.latency_ms)
    elif log.log_type == LogType.SYSTEM:
        assert isinstance(log.payload, SystemLog)
        features["cpu_pct"] = float(log.payload.cpu_pct)
        features["mem_pct"] = float(log.payload.mem_pct)
        features["disk_io_mb"] = float(log.payload.disk_io_mb)
    elif log.log_type == LogType.DB:
        assert isinstance(log.payload, DbLog)
        features["duration_ms"] = float(log.payload.duration_ms)
        features["rows_returned"] = float(log.payload.rows_returned)
    else:
        raise ValueError(f"Unsupported log type: {log.log_type}")

    return features

"""Log generation and anomaly injection for the ingestion service."""

import random
import time
import os
from datetime import datetime, timezone
from kafka import KafkaProducer
from typing import Literal, cast
from schemas import LogEntry, LogType, HttpLog, SystemLog, DbLog


HTTP_SOURCES = ["web-01", "web-02", "api-gateway"]
SYSTEM_SOURCES = ["worker-01", "worker-02", "worker-03", "ml-server-01"]
DB_SOURCES = ["postgres-primary", "postgres-replica-01"]
DB_QUERIES = [
    "SELECT * FROM users WHERE id = ?",
    "SELECT * FROM orders WHERE user_id = ?",
    "INSERT INTO orders (user_id, product_id, quantity) VALUES (?, ?, ?)",
    "UPDATE products SET stock = stock - 1 WHERE id = ?",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "SELECT COUNT(*) FROM events WHERE created_at > ?",
]


class AnomalyInjector:
    """Randomly injects anomalies into log entries at a configurable rate."""

    def __init__(self, rate: float = 0.05):
        """Initializes the injector with a given anomaly rate.

        Args:
            rate: Probability of injecting an anomaly into any given log entry.
        """
        self.rate = rate

    def inject(self, log: LogEntry) -> LogEntry:
        """Potentially injects an anomaly into a log entry.

        With probability self.rate, returns a modified copy of the log with
        is_anomaly set to True and one field altered to represent an anomaly.
        Otherwise returns the log unchanged.

        Args:
            log: The log entry to potentially modify.

        Returns:
            The original log entry, or a modified copy with an injected anomaly.
        """
        if random.random() > self.rate:
            return log

        if log.log_type == LogType.HTTP:
            return self._inject_http(log)
        elif log.log_type == LogType.SYSTEM:
            return self._inject_system(log)
        return self._inject_db(log)

    def _inject_http(self, log: LogEntry) -> LogEntry:
        """Injects a latency spike or 5xx status code into an HTTP log entry.

        Args:
            log: A log entry with an HttpLog payload.

        Returns:
            A modified copy with is_anomaly set to True.
        """
        assert isinstance(log.payload, HttpLog)
        anomaly_type = random.choice(["latency_spike", "5xx"])
        if anomaly_type == "latency_spike":
            new_payload = log.payload.model_copy(
                update={"latency_ms": log.payload.latency_ms * 10}
            )
        else:
            new_payload = log.payload.model_copy(
                update={"status_code": random.choice([500, 502, 503])}
            )
        return log.model_copy(update={"payload": new_payload, "is_anomaly": True})

    def _inject_system(self, log: LogEntry) -> LogEntry:
        """Injects a CPU spike or memory pressure anomaly into a system log entry.

        Args:
            log: A log entry with a SystemLog payload.

        Returns:
            A modified copy with is_anomaly set to True.
        """
        assert isinstance(log.payload, SystemLog)
        anomaly_type = random.choice(["cpu_spike", "memory_pressure"])
        if anomaly_type == "cpu_spike":
            new_payload = log.payload.model_copy(
                update={"cpu_pct": random.uniform(92, 99)}
            )
        else:
            new_payload = log.payload.model_copy(
                update={"mem_pct": random.uniform(90, 98)}
            )
        return log.model_copy(update={"payload": new_payload, "is_anomaly": True})

    def _inject_db(self, log: LogEntry) -> LogEntry:
        """Injects a slow query or zero rows anomaly into a DB log entry.

        Zero rows is only applied to SELECT queries. Non-SELECT queries
        always receive a slow query anomaly.

        Args:
            log: A log entry with a DbLog payload.

        Returns:
            A modified copy with is_anomaly set to True.
        """
        assert isinstance(log.payload, DbLog)
        anomaly_type = random.choice(["slow_query", "zero_rows"])
        if anomaly_type == "zero_rows" and log.payload.query.upper().startswith(
            "SELECT"
        ):
            new_payload = log.payload.model_copy(update={"rows_returned": 0})
        else:
            new_payload = log.payload.model_copy(
                update={"duration_ms": log.payload.duration_ms * 20}
            )
        return log.model_copy(update={"payload": new_payload, "is_anomaly": True})


class LogGenerator:
    """Generates synthetic log entries and publishes them to Kafka.

    Produces HTTP, system, and database logs drawn from realistic distributions.
    Anomaly injection is delegated to an AnomalyInjector instance.
    """

    def __init__(self, injector: AnomalyInjector, producer: KafkaProducer | None):
        """Initializes the generator with an injector and a Kafka producer.

        Args:
            injector: AnomalyInjector instance controlling anomaly rate and logic.
            producer: Kafka producer used to publish log entries to raw-logs.
                Pass None when using the generator without Kafka (e.g. dataset generation).
        """
        self.injector = injector
        self.producer = producer

    def generate_one_log(self) -> LogEntry:
        """Generates a single random log entry and passes it through the anomaly injector.

        Returns:
            A LogEntry, potentially modified with an injected anomaly.
        """
        log_type = random.choice([LogType.HTTP, LogType.SYSTEM, LogType.DB])
        if log_type == LogType.HTTP:
            log = self._generate_http()
        elif log_type == LogType.SYSTEM:
            log = self._generate_system()
        else:
            log = self._generate_db()
        return self.injector.inject(log)

    def run(self) -> None:
        """Runs the generation loop indefinitely.

        On each iteration, calls `generate_one_log` and publishes the generated log to the raw-logs Kafka topic.
        Sleeps for GENERATION_INTERVAL_MS milliseconds between iterations.
        """
        assert self.producer is not None
        while True:
            log = self.generate_one_log()
            self.producer.send("raw-logs", value=log.model_dump_json().encode("utf-8"))
            time.sleep(int(os.getenv("GENERATION_INTERVAL_MS", "500")) / 1000)

    def _generate_http(self) -> LogEntry:
        """Generates a synthetic HTTP log entry with realistic field distributions.

        Returns:
            A LogEntry with an HttpLog payload and is_anomaly set to False.
        """
        source = random.choice(HTTP_SOURCES)
        method = cast(
            Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
            random.choices(
                ["GET", "POST", "PUT", "DELETE", "PATCH"], weights=[60, 25, 8, 5, 2]
            )[0],
        )
        path = random.choice(
            ["/api/users", "/api/orders", "/health", "/api/products", "/api/auth"]
        )
        status_code = random.choices(
            [200, 201, 204, 400, 404, 500], weights=[70, 10, 5, 8, 5, 2]
        )[0]
        latency_ms = max(1.0, min(2000.0, random.normalvariate(150, 40)))
        user_agent = random.choice(
            [
                "Mozilla/5.0 (Chrome)",
                "Mozilla/5.0 (Firefox)",
                "Mozilla/5.0 (Safari)",
                "curl/7.68.0",
                "python-requests/2.28.0",
            ]
        )
        http_log = HttpLog(
            type="http",
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
            user_agent=user_agent,
        )
        log_entry = LogEntry(
            timestamp=datetime.now(tz=timezone.utc),
            log_type=LogType.HTTP,
            source=source,
            payload=http_log,
            is_anomaly=False,
        )
        return log_entry

    def _generate_system(self) -> LogEntry:
        """Generates a synthetic system log entry with realistic field distributions.

        Returns:
            A LogEntry with a SystemLog payload and is_anomaly set to False.
        """
        source = random.choice(SYSTEM_SOURCES)
        cpu_pct = max(0.0, min(100.0, random.normalvariate(45, 15)))
        mem_pct = max(0.0, min(100.0, random.normalvariate(60, 10)))
        disk_io_mb = max(0.0, random.normalvariate(50, 20))
        hostname = source
        system_log = SystemLog(
            type="system",
            cpu_pct=cpu_pct,
            mem_pct=mem_pct,
            disk_io_mb=disk_io_mb,
            hostname=hostname,
        )
        log_entry = LogEntry(
            timestamp=datetime.now(tz=timezone.utc),
            log_type=LogType.SYSTEM,
            source=source,
            payload=system_log,
            is_anomaly=False,
        )
        return log_entry

    def _generate_db(self) -> LogEntry:
        """Generates a synthetic database log entry with realistic field distributions.

        Returns:
            A LogEntry with a DbLog payload and is_anomaly set to False.
        """
        source = random.choice(DB_SOURCES)
        query = random.choice(DB_QUERIES)
        duration_ms = max(1.0, random.normalvariate(25, 10))
        rows_returned = int(max(0.0, random.normalvariate(10, 5)))
        db_name = source
        db_log = DbLog(
            type="db",
            query=query,
            duration_ms=duration_ms,
            rows_returned=rows_returned,
            db_name=db_name,
        )
        log_entry = LogEntry(
            timestamp=datetime.now(tz=timezone.utc),
            log_type=LogType.DB,
            source=source,
            payload=db_log,
            is_anomaly=False,
        )
        return log_entry

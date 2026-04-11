"""Kafka consumer for the ingestion service.

Polls raw log events from Kafka, validates them, and forwards
feature payloads to the ML service for inference.
"""

import json
import logging
import os

import httpx
from kafka import KafkaConsumer
from pydantic import ValidationError

from shared.schemas import DbLog, HttpLog, LogEntry, LogType, SystemLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogConsumer:
    """Consumes log entries from Kafka and forwards them to the ML service.

    Validates each message against the LogEntry schema before forwarding.
    Malformed messages and ML service errors are logged and skipped.
    """

    def __init__(self, consumer: KafkaConsumer, ml_service_url: str) -> None:
        """Initializes the consumer with a Kafka consumer and ML service URL.

        Args:
            consumer: Configured KafkaConsumer instance subscribed to raw-logs.
            ml_service_url: Base URL of the ML inference service.
        """
        self.consumer = consumer
        self.ml_service_url = ml_service_url

    def run(self) -> None:
        """Polls Kafka indefinitely and processes each incoming message.

        Skips and logs errors on malformed messages or ML service failures
        without interrupting the polling loop.
        """
        logger.info("Consumer started, polling raw-logs...")
        for message in self.consumer:
            self._process_message(message.value)

    def _process_message(self, raw: bytes) -> None:
        """Deserializes, validates, and forwards a single Kafka message.

        Args:
            raw: Raw bytes from the Kafka message value.
        """
        try:
            payload = json.loads(raw.decode("utf-8"))
            log = LogEntry.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Malformed message, skipping: %s", exc)
            return

        features = self._extract_features(log)
        body = {
            "log_type": log.log_type.value,
            "source": log.source,
            "timestamp": log.timestamp.isoformat(),
            "features": features,
        }

        try:
            response = httpx.post(
                f"{self.ml_service_url}/predict",
                json=body,
                timeout=5.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("ML service error, skipping: %s", exc)

    def _extract_features(self, log: LogEntry) -> dict:
        """Extracts raw feature fields from a log entry payload.

        Returns all payload fields as a flat dict. No transformation is
        applied — encoding and aggregation are handled by the ML service.

        Args:
            log: A validated log entry.

        Returns:
            A dict of raw feature fields for the log payload.
        """
        if log.log_type == LogType.HTTP:
            assert isinstance(log.payload, HttpLog)
            return {
                "method": log.payload.method,
                "path": log.payload.path,
                "status_code": log.payload.status_code,
                "latency_ms": log.payload.latency_ms,
                "user_agent": log.payload.user_agent,
            }
        if log.log_type == LogType.SYSTEM:
            assert isinstance(log.payload, SystemLog)
            return {
                "cpu_pct": log.payload.cpu_pct,
                "mem_pct": log.payload.mem_pct,
                "disk_io_mb": log.payload.disk_io_mb,
                "hostname": log.payload.hostname,
            }
        assert isinstance(log.payload, DbLog)
        return {
            "query": log.payload.query,
            "duration_ms": log.payload.duration_ms,
            "rows_returned": log.payload.rows_returned,
            "db_name": log.payload.db_name,
        }


if __name__ == "__main__":
    kafka_consumer = KafkaConsumer(
        "raw-logs",
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        value_deserializer=None,
        auto_offset_reset="earliest",
        group_id="ingestion-consumer",
    )
    consumer = LogConsumer(
        consumer=kafka_consumer,
        ml_service_url=os.getenv("ML_SERVICE_URL", "http://localhost:8000"),
    )
    consumer.run()

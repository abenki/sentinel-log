"""Pydantic schemas for log entries produced by the ingestion service."""

from typing import Annotated, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class LogType(str, Enum):
    HTTP = "http"
    SYSTEM = "system"
    DB = "db"


class HttpLog(BaseModel):
    type: Literal["http"]
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    latency_ms: float = Field(gt=0)
    user_agent: str = Field(min_length=1)


class SystemLog(BaseModel):
    type: Literal["system"]
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    disk_io_mb: float = Field(ge=0)
    hostname: str = Field(min_length=1)


class DbLog(BaseModel):
    type: Literal["db"]
    query: str = Field(min_length=1)
    duration_ms: float = Field(gt=0)
    rows_returned: int = Field(ge=0)
    db_name: str = Field(min_length=1)


class LogEntry(BaseModel):
    timestamp: datetime
    log_type: LogType
    source: str
    payload: Annotated[HttpLog | SystemLog | DbLog, Field(discriminator="type")]
    is_anomaly: bool

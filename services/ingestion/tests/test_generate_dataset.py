"""Tests for the dataset generation script."""

import json
from pathlib import Path

import pytest

from generate_dataset import main
from schemas import LogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_main(
    monkeypatch,
    tmp_path: Path,
    n_logs: int = 500,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> Path:
    """Runs main() with the given arguments and returns the output path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest tmp_path fixture.
        n_logs: Number of logs to generate.
        anomaly_rate: Anomaly injection rate.
        seed: Random seed.

    Returns:
        Path to the generated JSONL file.
    """
    output = tmp_path / "logs.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--n-logs",
            str(n_logs),
            "--anomaly-rate",
            str(anomaly_rate),
            "--output",
            str(output),
            "--seed",
            str(seed),
        ],
    )
    main()
    return output


# ---------------------------------------------------------------------------
# Output file creation
# ---------------------------------------------------------------------------


def test_output_file_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() creates the output file at the specified path."""
    output = run_main(monkeypatch, tmp_path)
    assert output.exists()


def test_parent_directories_are_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() creates intermediate parent directories if they do not exist."""
    output = tmp_path / "nested" / "dir" / "logs.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--n-logs",
            "100",
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    main()
    assert output.exists()


# ---------------------------------------------------------------------------
# Line count
# ---------------------------------------------------------------------------


def test_line_count_matches_n_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output file contains exactly n_logs non-empty lines."""
    output = run_main(monkeypatch, tmp_path, n_logs=500)
    lines = [
        line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 500


# ---------------------------------------------------------------------------
# JSON validity and schema
# ---------------------------------------------------------------------------


def test_each_line_is_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every line in the output file is valid JSON."""
    output = run_main(monkeypatch, tmp_path)
    for line in output.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # raises if invalid


def test_each_line_parses_as_log_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every line deserializes to a valid LogEntry without validation errors."""
    output = run_main(monkeypatch, tmp_path)
    for line in output.read_text(encoding="utf-8").splitlines():
        LogEntry.model_validate_json(line)


def test_is_anomaly_field_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every line contains the is_anomaly ground truth label."""
    output = run_main(monkeypatch, tmp_path)
    for line in output.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        assert "is_anomaly" in data
        assert isinstance(data["is_anomaly"], bool)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs with the same seed produce identical output files except for timestamps."""
    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "b.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--n-logs",
            "500",
            "--output",
            str(output_a),
            "--seed",
            "42",
        ],
    )
    main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--n-logs",
            "500",
            "--output",
            str(output_b),
            "--seed",
            "42",
        ],
    )
    main()

    def load_without_timestamp(path: Path) -> list[dict]:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        for row in rows:
            del row["timestamp"]
        return rows

    assert load_without_timestamp(output_a) == load_without_timestamp(output_b)


# ---------------------------------------------------------------------------
# Anomaly rate
# ---------------------------------------------------------------------------


def test_anomaly_rate_is_approximately_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed anomaly rate is within 2 percentage points of the configured rate."""
    output = run_main(monkeypatch, tmp_path, n_logs=2000, anomaly_rate=0.05)
    lines = output.read_text(encoding="utf-8").splitlines()
    anomaly_count = sum(1 for line in lines if json.loads(line)["is_anomaly"])
    observed_rate = anomaly_count / len(lines)
    assert abs(observed_rate - 0.05) < 0.02

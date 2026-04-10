"""Script to generate a labeled log dataset for ML training and evaluation."""

import argparse
import random
from pathlib import Path

from generator import AnomalyInjector, LogGenerator


def main() -> None:
    """Generates a JSONL dataset of synthetic log entries.

    Parses CLI arguments, generates n_logs log entries using LogGenerator,
    and writes each entry as a JSON line to the output file. The is_anomaly
    ground truth label is preserved in the output; it is the responsibility
    of the training code to exclude it from model inputs.

    Prints progress every 10 000 logs and a summary on completion.
    """
    parser = argparse.ArgumentParser(
        description="Generate a labeled log dataset for ML training and evaluation."
    )
    parser.add_argument(
        "--n-logs",
        type=int,
        default=60000,
        help="Total number of log entries to generate (default: 60000).",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Probability of injecting an anomaly into any given log entry (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../data/logs.jsonl",
        help="Path to the output JSONL file (default: data/logs.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    injector = AnomalyInjector(rate=args.anomaly_rate)
    generator = LogGenerator(injector=injector, producer=None)

    anomaly_count = 0

    print(f"Generating {args.n_logs} logs -> {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(args.n_logs):
            log = generator.generate_one_log()
            if log.is_anomaly:
                anomaly_count += 1
            f.write(log.model_dump_json() + "\n")
            if i > 0 and i % 10000 == 0:
                print(f"  {i} / {args.n_logs} logs written...")

    actual_rate = anomaly_count / args.n_logs if args.n_logs > 0 else 0.0
    print(f"Done. {args.n_logs} logs written to {output_path}")
    print(f"  anomalies : {anomaly_count} / {args.n_logs} ({actual_rate:.1%})")


if __name__ == "__main__":
    main()

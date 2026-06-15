from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "情感分析"
DEFAULT_INPUT = WORK_DIR / "outputs" / "semantic_annotations_for_review.csv"
DEFAULT_OUTPUT = WORK_DIR / "outputs" / "semantic_annotations_priority_review.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def add_reason(row: dict[str, str], reason: str) -> dict[str, str]:
    copied = dict(row)
    existing = copied.get("review_priority_reason", "")
    copied["review_priority_reason"] = reason if not existing else f"{existing};{reason}"
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a priority file for human review.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-liked", type=int, default=120)
    parser.add_argument("--top-replied", type=int, default=80)
    parser.add_argument("--random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rows = read_csv(args.input)
    selected: dict[str, dict[str, str]] = {}

    def put(row: dict[str, str], reason: str) -> None:
        content_id = row["content_id"]
        if content_id in selected:
            selected[content_id] = add_reason(selected[content_id], reason)
        else:
            selected[content_id] = add_reason(row, reason)

    for row in rows:
        if row.get("need_review", "").lower() == "true":
            put(row, "need_review")
        if to_float(row.get("confidence", "")) < 0.70:
            put(row, "low_confidence")
        risk_labels = row.get("risk_labels", "")
        if risk_labels and risk_labels != "low_risk_discussion":
            put(row, "risk_label")

    for row in sorted(rows, key=lambda item: to_int(item.get("like_count", "")), reverse=True)[: args.top_liked]:
        put(row, "top_liked")

    for row in sorted(rows, key=lambda item: to_int(item.get("received_reply_count", "")), reverse=True)[
        : args.top_replied
    ]:
        put(row, "top_replied")

    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    for row in shuffled[: args.random]:
        put(row, "random_check")

    priority_rows = list(selected.values())
    priority_rows.sort(
        key=lambda item: (
            "need_review" not in item.get("review_priority_reason", ""),
            -to_int(item.get("like_count", "")),
            -to_int(item.get("received_reply_count", "")),
        )
    )

    fieldnames = ["review_priority_reason"] + [field for field in rows[0].keys() if field != "review_priority_reason"]
    write_csv(args.output, priority_rows, fieldnames)
    print(f"Wrote {len(priority_rows)} priority review rows to {args.output}")


if __name__ == "__main__":
    main()

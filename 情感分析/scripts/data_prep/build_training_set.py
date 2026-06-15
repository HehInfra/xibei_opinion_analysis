from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


def find_work_dir() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if path.name == "情感分析":
            return path
    raise RuntimeError("无法定位 情感分析 工作目录")


WORK_DIR = find_work_dir()
DEFAULT_INPUT = WORK_DIR / "outputs" / "annotations" / "semantic_annotations_for_review.csv"
OUT_DIR = WORK_DIR / "data" / "training"
TRAIN_PATH = OUT_DIR / "semantic_train.csv"
SUMMARY_PATH = OUT_DIR / "semantic_train_summary.md"


TASKS = {
    "topic": ["content_id", "content_text", "topic_label"],
    "stance_target": ["content_id", "content_text", "stance_target"],
    "stance": ["content_id", "content_text", "stance_label"],
    "emotion": ["content_id", "content_text", "emotion_label"],
    "discourse": ["content_id", "content_text", "discourse_labels"],
    "risk": ["content_id", "content_text", "risk_labels"],
}


REQUIRED_FIELDS = [
    "content_id",
    "content_text",
    "topic_label",
    "stance_target",
    "stance_label",
    "emotion_label",
    "discourse_labels",
    "risk_labels",
]


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


def is_complete(row: dict[str, str]) -> bool:
    return all(row.get(field, "").strip() for field in REQUIRED_FIELDS)


def split_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = round(n * 0.8)
    valid_end = round(n * 0.9)
    result = []
    for index, row in enumerate(shuffled):
        copied = dict(row)
        if index < train_end:
            copied["split"] = "train"
        elif index < valid_end:
            copied["split"] = "valid"
        else:
            copied["split"] = "test"
        result.append(copied)
    return result


def write_task_files(rows: list[dict[str, str]]) -> None:
    for task_name, fields in TASKS.items():
        task_rows = [{field: row.get(field, "") for field in fields + ["split"]} for row in rows]
        write_csv(OUT_DIR / f"{task_name}.csv", task_rows, fields + ["split"])


def write_summary(rows: list[dict[str, str]], source_path: Path) -> None:
    lines = [
        "# 语义训练集摘要",
        "",
        f"- 来源文件：`{source_path}`",
        f"- 可用样本数：{len(rows)}",
        "",
        "## 数据划分",
        "",
    ]
    for split, count in Counter(row["split"] for row in rows).most_common():
        lines.append(f"- {split}：{count}")

    for field in ["topic_label", "stance_target", "stance_label", "emotion_label"]:
        lines.extend(["", f"## {field}", ""])
        for label, count in Counter(row[field] for row in rows).most_common():
            lines.append(f"- {label}：{count}")

    for field in ["discourse_labels", "risk_labels"]:
        lines.extend(["", f"## {field}", ""])
        counter: Counter[str] = Counter()
        for row in rows:
            labels = [item.strip() for item in row[field].replace(",", ";").split(";") if item.strip()]
            counter.update(labels)
        for label, count in counter.most_common():
            lines.append(f"- {label}：{count}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/valid/test files from reviewed semantic annotations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--include-review", action="store_true", help="Include rows still marked need_review=true.")
    args = parser.parse_args()

    rows = read_csv(args.input)
    complete_rows = [row for row in rows if is_complete(row)]
    if not args.include_review:
        complete_rows = [row for row in complete_rows if row.get("need_review", "").lower() != "true"]

    if not complete_rows:
        raise SystemExit("No complete reviewed rows found. Finish AI annotation and human review first.")

    split = split_rows(complete_rows, args.seed)
    fields = list(split[0].keys())
    write_csv(TRAIN_PATH, split, fields)
    write_task_files(split)
    write_summary(split, args.input)
    print(f"Wrote {len(split)} rows to {TRAIN_PATH}")


if __name__ == "__main__":
    main()

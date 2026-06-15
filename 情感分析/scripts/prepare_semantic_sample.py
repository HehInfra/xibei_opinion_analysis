from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "情感分析"
DATA_DIR = WORK_DIR / "data"
DB_PATH = ROOT / "data" / "database" / "xibei_event.db"

FULL_COMMENTS_PATH = DATA_DIR / "semantic_comments_full.csv"
SAMPLE_PATH = DATA_DIR / "semantic_annotation_sample.csv"
REVIEW_TEMPLATE_PATH = DATA_DIR / "semantic_annotation_review_template.csv"
SUMMARY_PATH = DATA_DIR / "semantic_sample_summary.md"


ANNOTATION_FIELDS = [
    "topic_label",
    "stance_target",
    "stance_label",
    "emotion_label",
    "discourse_labels",
    "risk_labels",
    "intensity",
    "confidence",
    "need_review",
    "annotation_reason",
]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_comments(conn: sqlite3.Connection) -> list[dict[str, str]]:
    query = """
    WITH reply_counts AS (
        SELECT target_content_id AS content_id, COUNT(*) AS received_reply_count
        FROM relations
        WHERE relation_type = 'reply'
        GROUP BY target_content_id
    )
    SELECT
        c.content_id,
        c.raw_content_id,
        c.source_id,
        s.source_title,
        s.platform_source_id,
        s.source_url,
        s.author_user_id,
        s.published_at AS source_published_at,
        c.content_type,
        c.user_id,
        u.user_name,
        u.gender,
        u.level,
        c.content_text,
        c.created_at,
        CAST(COALESCE(NULLIF(c.like_count, ''), '0') AS INTEGER) AS like_count,
        c.parent_content_id,
        c.root_content_id,
        COALESCE(rc.received_reply_count, 0) AS received_reply_count,
        c.raw_file_path
    FROM contents c
    LEFT JOIN sources s ON c.source_id = s.source_id
    LEFT JOIN users u ON c.user_id = u.user_id
    LEFT JOIN reply_counts rc ON c.content_id = rc.content_id
    WHERE c.content_text IS NOT NULL AND TRIM(c.content_text) != ''
    ORDER BY c.created_at, c.content_id
    """
    rows = []
    for row in conn.execute(query):
        rows.append({key: "" if row[key] is None else str(row[key]) for key in row.keys()})
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fieldnames})


def clean_cell(value: str) -> str:
    return str(value).replace("\x00", "")


def add_rows(
    selected: dict[str, dict[str, str]],
    candidates: list[dict[str, str]],
    limit: int,
    bucket: str,
) -> None:
    count = 0
    for row in candidates:
        content_id = row["content_id"]
        if content_id in selected:
            continue
        copied = dict(row)
        copied["sample_bucket"] = bucket
        selected[content_id] = copied
        count += 1
        if count >= limit:
            break


def stratified_by_source(
    rows: list[dict[str, str]],
    *,
    per_source: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    result = []
    for source_rows in by_source.values():
        shuffled = list(source_rows)
        rng.shuffle(shuffled)
        result.extend(shuffled[:per_source])
    rng.shuffle(result)
    return result


def build_sample(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected: dict[str, dict[str, str]] = {}

    top_liked_limit = round(sample_size * 0.20)
    top_replied_limit = round(sample_size * 0.15)
    stratified_limit = round(sample_size * 0.25)

    top_liked = sorted(rows, key=lambda row: int(row["like_count"]), reverse=True)
    add_rows(selected, top_liked, top_liked_limit, "top_liked")

    top_replied = sorted(rows, key=lambda row: int(row["received_reply_count"]), reverse=True)
    add_rows(selected, top_replied, top_replied_limit, "top_replied")

    per_source = max(1, stratified_limit // max(1, len({row["source_id"] for row in rows})))
    source_candidates = stratified_by_source(rows, per_source=per_source, seed=seed + 1)
    add_rows(selected, source_candidates, stratified_limit, "source_stratified")

    remaining = list(rows)
    rng.shuffle(remaining)
    add_rows(selected, remaining, sample_size - len(selected), "random")

    sample = list(selected.values())[:sample_size]
    for index, row in enumerate(sample, start=1):
        row["sample_id"] = f"S{index:04d}"
    return sample


def write_summary(rows: list[dict[str, str]], sample: list[dict[str, str]], sample_size: int, seed: int) -> None:
    source_count = len({row["source_id"] for row in rows})
    user_count = len({row["user_id"] for row in rows})
    content_type_counts = Counter(row["content_type"] for row in rows)
    sample_bucket_counts = Counter(row["sample_bucket"] for row in sample)
    sample_type_counts = Counter(row["content_type"] for row in sample)
    top_sources = Counter(row["source_title"] for row in sample).most_common(10)

    text = [
        "# 语义标注抽样摘要",
        "",
        "## 全量评论数据",
        "",
        f"- 评论总数：{len(rows)}",
        f"- 视频源数：{source_count}",
        f"- 用户数：{user_count}",
        f"- 一级评论数：{content_type_counts.get('comment', 0)}",
        f"- 楼中楼回复数：{content_type_counts.get('reply', 0)}",
        "",
        "## 抽样设置",
        "",
        f"- 目标样本数：{sample_size}",
        f"- 实际样本数：{len(sample)}",
        f"- 随机种子：{seed}",
        "- 抽样策略：高赞评论 + 高回复评论 + 视频分层随机 + 全局随机",
        "",
        "## 样本构成",
        "",
    ]
    for bucket, count in sample_bucket_counts.most_common():
        text.append(f"- {bucket}：{count}")
    text.extend(["", "## 评论层级", ""])
    for content_type, count in sample_type_counts.most_common():
        text.append(f"- {content_type}：{count}")
    text.extend(["", "## 样本 Top 视频来源", ""])
    for title, count in top_sources:
        text.append(f"- {title}：{count}")
    text.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 全量评论导出：`{FULL_COMMENTS_PATH.relative_to(WORK_DIR)}`",
            f"- 标注样本：`{SAMPLE_PATH.relative_to(WORK_DIR)}`",
            f"- 人工审核模板：`{REVIEW_TEMPLATE_PATH.relative_to(WORK_DIR)}`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare semantic annotation sample for Xibei Bilibili comments.")
    parser.add_argument("--sample-size", type=int, default=1500, help="Number of comments to sample.")
    parser.add_argument("--seed", type=int, default=20260614, help="Random seed.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with connect() as conn:
        rows = fetch_comments(conn)

    if args.sample_size > len(rows):
        raise ValueError(f"sample size {args.sample_size} exceeds row count {len(rows)}")

    base_fields = list(rows[0].keys())
    write_csv(FULL_COMMENTS_PATH, rows, base_fields)

    sample = build_sample(rows, args.sample_size, args.seed)
    sample_fields = ["sample_id", "sample_bucket"] + base_fields
    write_csv(SAMPLE_PATH, sample, sample_fields)

    review_rows = []
    for row in sample:
        review_row = dict(row)
        for field in ANNOTATION_FIELDS:
            review_row[field] = ""
        review_rows.append(review_row)
    write_csv(REVIEW_TEMPLATE_PATH, review_rows, sample_fields + ANNOTATION_FIELDS)

    write_summary(rows, sample, args.sample_size, args.seed)
    print(f"Exported {len(rows)} comments to {FULL_COMMENTS_PATH}")
    print(f"Sampled {len(sample)} comments to {SAMPLE_PATH}")
    print(f"Wrote review template to {REVIEW_TEMPLATE_PATH}")


if __name__ == "__main__":
    main()

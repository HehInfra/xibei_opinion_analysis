from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "xibei_event.db"
ANALYSIS_DIR = BASE_DIR / "analysis"
MODULE_DIR = ANALYSIS_DIR / "module1_overview"


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    return conn.execute(sql, params).fetchone()[0]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator * 100), 2) if denominator else 0.0


def main() -> int:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    source_count = scalar(conn, "select count(*) from sources")
    user_count = scalar(conn, "select count(*) from users")
    content_count = scalar(conn, "select count(*) from contents")
    relation_count = scalar(conn, "select count(*) from relations")
    raw_file_count = scalar(conn, "select count(*) from raw_file_audit")
    parsed_raw_file_count = scalar(conn, "select count(*) from raw_file_audit where can_extract_comment_json = 'true'")
    empty_raw_file_count = scalar(conn, "select count(*) from raw_file_audit where is_empty = 'true'")

    content_type_rows = rows(conn, "select content_type, count(*) as count from contents group by content_type order by count desc")
    relation_type_rows = rows(conn, "select relation_type, count(*) as count from relations group by relation_type order by count desc")

    time_bounds = rows(
        conn,
        """
        select min(created_at) as min_created_at, max(created_at) as max_created_at
        from contents
        where created_at is not null and created_at != ''
        """,
    )[0]

    missing_quality = {
        "sources_missing_title": scalar(conn, "select count(*) from sources where source_title is null or source_title = ''"),
        "sources_missing_published_at": scalar(conn, "select count(*) from sources where published_at is null or published_at = ''"),
        "sources_missing_url": scalar(conn, "select count(*) from sources where source_url is null or source_url = ''"),
        "contents_missing_text": scalar(conn, "select count(*) from contents where content_text is null or content_text = ''"),
        "contents_missing_created_at": scalar(conn, "select count(*) from contents where created_at is null or created_at = ''"),
        "contents_missing_user": scalar(conn, "select count(*) from contents where user_id is null or user_id = ''"),
        "relations_missing_source_user": scalar(conn, "select count(*) from relations where source_user_id is null or source_user_id = ''"),
        "reply_relations_missing_target_user": scalar(conn, "select count(*) from relations where relation_type = 'reply' and (target_user_id is null or target_user_id = '')"),
    }

    video_summary = rows(
        conn,
        """
        select
          s.source_id,
          s.source_title,
          s.published_at,
          cast(nullif(s.comment_all_count, '') as integer) as comment_all_count,
          cast(nullif(s.page_reply_count, '') as integer) as page_reply_count,
          s.has_next_offset,
          count(c.content_id) as extracted_content_count,
          sum(case when c.content_type = 'comment' then 1 else 0 end) as extracted_comment_count,
          sum(case when c.content_type = 'reply' then 1 else 0 end) as extracted_reply_count,
          count(distinct c.user_id) as distinct_user_count,
          coalesce(sum(cast(nullif(c.like_count, '') as integer)), 0) as extracted_like_sum,
          max(cast(nullif(c.like_count, '') as integer)) as max_like_count,
          min(c.created_at) as first_comment_at,
          max(c.created_at) as last_comment_at
        from sources s
        left join contents c on c.source_id = s.source_id
        group by s.source_id
        order by comment_all_count desc, extracted_content_count desc
        """,
    )
    for row in video_summary:
        row["sample_coverage_percent"] = pct(row.get("extracted_comment_count") or 0, row.get("comment_all_count") or 0)

    raw_audit = rows(
        conn,
        """
        select
          file_name,
          file_size,
          is_empty,
          can_extract_comment_json,
          has_video_meta,
          video_name,
          comment_all_count,
          page_reply_count,
          has_next_offset,
          nested_reply_count,
          error_message
        from raw_file_audit
        order by can_extract_comment_json asc, file_name asc
        """,
    )

    user_presence = rows(
        conn,
        """
        select
          u.user_id,
          u.user_name,
          count(c.content_id) as content_count,
          count(distinct c.source_id) as source_count,
          min(c.created_at) as first_seen_at,
          max(c.created_at) as last_seen_at,
          coalesce(sum(cast(nullif(c.like_count, '') as integer)), 0) as received_like_sum
        from users u
        left join contents c on c.user_id = u.user_id
        group by u.user_id
        order by content_count desc, received_like_sum desc
        limit 50
        """,
    )

    time_distribution_rows = rows(conn, "select created_at, content_type from contents where created_at is not null and created_at != ''")
    daily_counter: dict[str, Counter[str]] = defaultdict(Counter)
    hourly_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in time_distribution_rows:
        dt = parse_dt(row["created_at"])
        if not dt:
            continue
        day = dt.date().isoformat()
        hour = dt.strftime("%Y-%m-%d %H:00")
        daily_counter[day][row["content_type"]] += 1
        daily_counter[day]["total"] += 1
        hourly_counter[hour][row["content_type"]] += 1
        hourly_counter[hour]["total"] += 1

    daily_distribution = [
        {
            "date": key,
            "total": counter["total"],
            "comment": counter["comment"],
            "reply": counter["reply"],
        }
        for key, counter in sorted(daily_counter.items())
    ]
    hourly_distribution = [
        {
            "hour": key,
            "total": counter["total"],
            "comment": counter["comment"],
            "reply": counter["reply"],
        }
        for key, counter in sorted(hourly_counter.items())
    ]

    overview = {
        "module": "module1_overview",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_database": str(DB_PATH),
        "counts": {
            "raw_files": raw_file_count,
            "parsed_raw_files": parsed_raw_file_count,
            "empty_raw_files": empty_raw_file_count,
            "sources": source_count,
            "users": user_count,
            "contents": content_count,
            "relations": relation_count,
        },
        "content_types": content_type_rows,
        "relation_types": relation_type_rows,
        "time_range": time_bounds,
        "missing_quality": {
            **missing_quality,
            "contents_missing_text_percent": pct(missing_quality["contents_missing_text"], content_count),
            "contents_missing_created_at_percent": pct(missing_quality["contents_missing_created_at"], content_count),
            "contents_missing_user_percent": pct(missing_quality["contents_missing_user"], content_count),
        },
        "top_video_by_comment_all_count": video_summary[:5],
        "notes": [
            "当前数据来自 B 站视频评论 raw 文件。",
            "source_url 目前为空，说明还需要补充 BV 号或视频链接。",
            "comment_all_count 是平台返回的完整评论总数，extracted_content_count 是当前已解析样本量。",
        ],
    }

    write_json(MODULE_DIR / "overview.json", overview)
    write_csv(
        MODULE_DIR / "video_summary.csv",
        video_summary,
        [
            "source_id",
            "source_title",
            "published_at",
            "comment_all_count",
            "page_reply_count",
            "has_next_offset",
            "extracted_content_count",
            "extracted_comment_count",
            "extracted_reply_count",
            "distinct_user_count",
            "extracted_like_sum",
            "max_like_count",
            "first_comment_at",
            "last_comment_at",
            "sample_coverage_percent",
        ],
    )
    write_csv(
        MODULE_DIR / "raw_audit_summary.csv",
        raw_audit,
        [
            "file_name",
            "file_size",
            "is_empty",
            "can_extract_comment_json",
            "has_video_meta",
            "video_name",
            "comment_all_count",
            "page_reply_count",
            "has_next_offset",
            "nested_reply_count",
            "error_message",
        ],
    )
    write_csv(
        MODULE_DIR / "user_presence_top50.csv",
        user_presence,
        ["user_id", "user_name", "content_count", "source_count", "first_seen_at", "last_seen_at", "received_like_sum"],
    )
    write_csv(MODULE_DIR / "daily_distribution.csv", daily_distribution, ["date", "total", "comment", "reply"])
    write_csv(MODULE_DIR / "hourly_distribution.csv", hourly_distribution, ["hour", "total", "comment", "reply"])

    summary_md = f"""# 模块 1：数据概览

## 核心统计

```text
raw 文件数：{raw_file_count}
成功解析 raw 文件数：{parsed_raw_file_count}
空 raw 文件数：{empty_raw_file_count}
视频源数：{source_count}
用户数：{user_count}
内容数：{content_count}
关系数：{relation_count}
```

## 内容结构

```text
一级评论：{next((r['count'] for r in content_type_rows if r['content_type'] == 'comment'), 0)}
楼中楼回复：{next((r['count'] for r in content_type_rows if r['content_type'] == 'reply'), 0)}
```

## 关系结构

```text
用户评论视频 comment_source：{next((r['count'] for r in relation_type_rows if r['relation_type'] == 'comment_source'), 0)}
用户回复用户 reply：{next((r['count'] for r in relation_type_rows if r['relation_type'] == 'reply'), 0)}
```

## 时间范围

```text
最早评论时间：{time_bounds.get('min_created_at') or ''}
最晚评论时间：{time_bounds.get('max_created_at') or ''}
```

## 数据质量观察

```text
缺失正文内容数：{missing_quality['contents_missing_text']}
缺失发布时间内容数：{missing_quality['contents_missing_created_at']}
缺失用户内容数：{missing_quality['contents_missing_user']}
缺失 source_url 的视频源数：{missing_quality['sources_missing_url']}
```

当前数据已经可以支撑后续模块的基础分析，包括热度分析、主题分析、情绪立场分析和回复网络分析。需要注意的是，当前 `source_url` 仍为空，说明后续如果要做可回溯展示，最好补充 BV 号或视频链接。
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"overview: {MODULE_DIR / 'overview.json'}")
    print(f"video_summary: {MODULE_DIR / 'video_summary.csv'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


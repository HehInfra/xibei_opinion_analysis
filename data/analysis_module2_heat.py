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
MODULE_DIR = BASE_DIR / "analysis" / "module2_heat"


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


def shorten(text: str, limit: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def main() -> int:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    content_count = scalar(conn, "select count(*) from contents")
    total_likes = scalar(conn, "select coalesce(sum(cast(nullif(like_count, '') as integer)), 0) from contents")
    positive_like_count = scalar(conn, "select count(*) from contents where cast(nullif(like_count, '') as integer) > 0")
    max_like = scalar(conn, "select max(cast(nullif(like_count, '') as integer)) from contents")
    avg_like = scalar(conn, "select avg(cast(nullif(like_count, '') as integer)) from contents")

    video_heat = rows(
        conn,
        """
        select
          s.source_id,
          s.source_title,
          s.published_at,
          cast(nullif(s.comment_all_count, '') as integer) as platform_comment_count,
          count(c.content_id) as extracted_content_count,
          sum(case when c.content_type = 'comment' then 1 else 0 end) as extracted_comment_count,
          sum(case when c.content_type = 'reply' then 1 else 0 end) as extracted_reply_count,
          count(distinct c.user_id) as distinct_user_count,
          coalesce(sum(cast(nullif(c.like_count, '') as integer)), 0) as extracted_like_sum,
          max(cast(nullif(c.like_count, '') as integer)) as max_comment_like,
          round(avg(cast(nullif(c.like_count, '') as integer)), 2) as avg_comment_like,
          min(c.created_at) as first_comment_at,
          max(c.created_at) as last_comment_at
        from sources s
        left join contents c on c.source_id = s.source_id
        group by s.source_id
        order by extracted_like_sum desc, platform_comment_count desc
        """,
    )
    for row in video_heat:
        platform_count = row.get("platform_comment_count") or 0
        extracted_comments = row.get("extracted_comment_count") or 0
        row["sample_coverage_percent"] = pct(extracted_comments, platform_count)
        row["heat_score"] = round(
            (row.get("extracted_like_sum") or 0)
            + (row.get("extracted_reply_count") or 0) * 20
            + (row.get("distinct_user_count") or 0) * 5,
            2,
        )

    hot_comments = rows(
        conn,
        """
        select
          c.content_id,
          c.content_type,
          c.source_id,
          s.source_title,
          c.user_id,
          u.user_name,
          c.content_text,
          c.created_at,
          cast(nullif(c.like_count, '') as integer) as like_count,
          c.parent_content_id,
          c.root_content_id
        from contents c
        left join sources s on s.source_id = c.source_id
        left join users u on u.user_id = c.user_id
        order by like_count desc
        limit 100
        """,
    )
    for rank, row in enumerate(hot_comments, start=1):
        row["rank"] = rank
        row["content_preview"] = shorten(row.get("content_text") or "", 120)

    active_users = rows(
        conn,
        """
        select
          u.user_id,
          u.user_name,
          count(c.content_id) as content_count,
          sum(case when c.content_type = 'comment' then 1 else 0 end) as comment_count,
          sum(case when c.content_type = 'reply' then 1 else 0 end) as reply_count,
          count(distinct c.source_id) as source_count,
          coalesce(sum(cast(nullif(c.like_count, '') as integer)), 0) as received_like_sum,
          max(cast(nullif(c.like_count, '') as integer)) as max_single_like,
          min(c.created_at) as first_seen_at,
          max(c.created_at) as last_seen_at
        from users u
        join contents c on c.user_id = u.user_id
        group by u.user_id
        order by content_count desc, source_count desc, received_like_sum desc
        limit 100
        """,
    )

    liked_users = rows(
        conn,
        """
        select
          u.user_id,
          u.user_name,
          count(c.content_id) as content_count,
          count(distinct c.source_id) as source_count,
          coalesce(sum(cast(nullif(c.like_count, '') as integer)), 0) as received_like_sum,
          max(cast(nullif(c.like_count, '') as integer)) as max_single_like,
          round(avg(cast(nullif(c.like_count, '') as integer)), 2) as avg_like,
          min(c.created_at) as first_seen_at,
          max(c.created_at) as last_seen_at
        from users u
        join contents c on c.user_id = u.user_id
        group by u.user_id
        order by received_like_sum desc, max_single_like desc
        limit 100
        """,
    )

    reply_targets = rows(
        conn,
        """
        select
          r.target_user_id as user_id,
          u.user_name,
          count(*) as received_reply_count,
          count(distinct r.source_user_id) as distinct_replier_count,
          count(distinct r.source_id) as source_count
        from relations r
        left join users u on u.user_id = r.target_user_id
        where r.relation_type = 'reply' and r.target_user_id != ''
        group by r.target_user_id
        order by received_reply_count desc, distinct_replier_count desc
        limit 100
        """,
    )

    time_rows = rows(
        conn,
        """
        select created_at, content_type, cast(nullif(like_count, '') as integer) as like_count
        from contents
        where created_at is not null and created_at != ''
        """,
    )
    daily: dict[str, Counter[str]] = defaultdict(Counter)
    hourly: dict[str, Counter[str]] = defaultdict(Counter)
    for row in time_rows:
        dt = parse_dt(row["created_at"])
        if not dt:
            continue
        like_count = row.get("like_count") or 0
        for key, bucket in ((dt.date().isoformat(), daily), (dt.strftime("%Y-%m-%d %H:00"), hourly)):
            bucket[key]["content_count"] += 1
            bucket[key][f"{row['content_type']}_count"] += 1
            bucket[key]["like_sum"] += like_count
            bucket[key]["max_like"] = max(bucket[key]["max_like"], like_count)

    daily_heat = [
        {
            "date": key,
            "content_count": counter["content_count"],
            "comment_count": counter["comment_count"],
            "reply_count": counter["reply_count"],
            "like_sum": counter["like_sum"],
            "max_like": counter["max_like"],
        }
        for key, counter in sorted(daily.items())
    ]
    hourly_heat = [
        {
            "hour": key,
            "content_count": counter["content_count"],
            "comment_count": counter["comment_count"],
            "reply_count": counter["reply_count"],
            "like_sum": counter["like_sum"],
            "max_like": counter["max_like"],
        }
        for key, counter in sorted(hourly.items())
    ]

    overview = {
        "module": "module2_heat",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_database": str(DB_PATH),
        "summary": {
            "content_count": content_count,
            "total_likes": total_likes,
            "positive_like_content_count": positive_like_count,
            "positive_like_content_percent": pct(positive_like_count, content_count),
            "max_like": max_like,
            "avg_like": round(avg_like or 0, 2),
        },
        "top_videos_by_heat_score": video_heat[:10],
        "top_comments_by_like": hot_comments[:20],
        "top_active_users": active_users[:20],
        "top_liked_users": liked_users[:20],
        "top_reply_targets": reply_targets[:20],
        "notes": [
            "热度分析基于当前已抽取样本，不等于平台完整评论区热度。",
            "video_heat 中 platform_comment_count 来自 B 站 cursor.all_count，extracted_* 来自当前样本。",
            "heat_score 是课程设计用综合分：样本获赞总数 + 楼中楼回复数*20 + 参与用户数*5。",
        ],
    }

    write_json(MODULE_DIR / "heat_overview.json", overview)
    write_csv(
        MODULE_DIR / "video_heat.csv",
        video_heat,
        [
            "source_id",
            "source_title",
            "published_at",
            "platform_comment_count",
            "extracted_content_count",
            "extracted_comment_count",
            "extracted_reply_count",
            "distinct_user_count",
            "extracted_like_sum",
            "max_comment_like",
            "avg_comment_like",
            "first_comment_at",
            "last_comment_at",
            "sample_coverage_percent",
            "heat_score",
        ],
    )
    write_csv(
        MODULE_DIR / "hot_comments_top100.csv",
        hot_comments,
        [
            "rank",
            "content_id",
            "content_type",
            "source_id",
            "source_title",
            "user_id",
            "user_name",
            "content_preview",
            "content_text",
            "created_at",
            "like_count",
            "parent_content_id",
            "root_content_id",
        ],
    )
    write_csv(
        MODULE_DIR / "active_users_top100.csv",
        active_users,
        [
            "user_id",
            "user_name",
            "content_count",
            "comment_count",
            "reply_count",
            "source_count",
            "received_like_sum",
            "max_single_like",
            "first_seen_at",
            "last_seen_at",
        ],
    )
    write_csv(
        MODULE_DIR / "liked_users_top100.csv",
        liked_users,
        [
            "user_id",
            "user_name",
            "content_count",
            "source_count",
            "received_like_sum",
            "max_single_like",
            "avg_like",
            "first_seen_at",
            "last_seen_at",
        ],
    )
    write_csv(
        MODULE_DIR / "reply_targets_top100.csv",
        reply_targets,
        ["user_id", "user_name", "received_reply_count", "distinct_replier_count", "source_count"],
    )
    write_csv(MODULE_DIR / "daily_heat.csv", daily_heat, ["date", "content_count", "comment_count", "reply_count", "like_sum", "max_like"])
    write_csv(MODULE_DIR / "hourly_heat.csv", hourly_heat, ["hour", "content_count", "comment_count", "reply_count", "like_sum", "max_like"])

    top_video = video_heat[0] if video_heat else {}
    top_comment = hot_comments[0] if hot_comments else {}
    top_active_user = active_users[0] if active_users else {}
    top_liked_user = liked_users[0] if liked_users else {}
    top_reply_target = reply_targets[0] if reply_targets else {}

    summary_md = f"""# 模块 2：热度分析

## 整体热度

```text
样本内容数：{content_count}
样本获赞总数：{total_likes}
有点赞内容数：{positive_like_count}
有点赞内容占比：{pct(positive_like_count, content_count)}%
单条最高点赞：{max_like}
平均点赞：{round(avg_like or 0, 2)}
```

## 最热视频

```text
视频标题：{top_video.get('source_title', '')}
平台评论总数：{top_video.get('platform_comment_count', '')}
样本内容数：{top_video.get('extracted_content_count', '')}
样本获赞总数：{top_video.get('extracted_like_sum', '')}
最高单条点赞：{top_video.get('max_comment_like', '')}
热度分：{top_video.get('heat_score', '')}
```

## 最高赞评论

```text
用户：{top_comment.get('user_name', '')}
点赞数：{top_comment.get('like_count', '')}
视频：{top_comment.get('source_title', '')}
评论：{shorten(top_comment.get('content_text', ''), 160)}
```

## 最活跃用户

```text
用户：{top_active_user.get('user_name', '')}
内容数：{top_active_user.get('content_count', '')}
参与视频数：{top_active_user.get('source_count', '')}
获赞总数：{top_active_user.get('received_like_sum', '')}
```

## 最高获赞用户

```text
用户：{top_liked_user.get('user_name', '')}
内容数：{top_liked_user.get('content_count', '')}
获赞总数：{top_liked_user.get('received_like_sum', '')}
最高单条点赞：{top_liked_user.get('max_single_like', '')}
```

## 被回复最多用户

```text
用户：{top_reply_target.get('user_name', '')}
收到回复数：{top_reply_target.get('received_reply_count', '')}
不同回复者数：{top_reply_target.get('distinct_replier_count', '')}
```

## 注意事项

当前热度分析基于已采样评论，不代表平台完整评论区的完整热度。尤其是部分视频平台评论总数很高，但当前只抽取了热门评论页，因此这些结果更适合解释“热门样本中的注意力分布”。
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"heat_overview: {MODULE_DIR / 'heat_overview.json'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    print(f"video_heat: {MODULE_DIR / 'video_heat.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


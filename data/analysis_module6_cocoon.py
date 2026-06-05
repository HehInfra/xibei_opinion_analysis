from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODULE2_HOT_COMMENTS = BASE_DIR / "analysis" / "module2_heat" / "hot_comments_top100.csv"
MODULE4_CONTENT_LABELS = BASE_DIR / "analysis" / "module4_sentiment_stance" / "content_sentiment_stance.csv"
MODULE5_REPLY_EDGES = BASE_DIR / "analysis" / "module5_network" / "reply_edges.csv"
MODULE_DIR = BASE_DIR / "analysis" / "module6_cocoon"


TEXT_WEIGHTS = {
    "topic_concentration": 0.30,
    "sentiment_consistency": 0.30,
    "stance_consistency": 0.40,
}

FULL_WEIGHTS = {
    "topic_concentration": 0.25,
    "sentiment_consistency": 0.25,
    "stance_consistency": 0.30,
    "interaction_homophily": 0.20,
}

MIN_REPLY_EDGES_FOR_STRUCTURE = 5


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(n: int | float, total: int | float) -> float:
    return round(n / total * 100, 2) if total else 0.0


def ratio(n: int | float, total: int | float) -> float:
    return round(n / total, 4) if total else 0.0


def dominant(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return "", 0
    return counter.most_common(1)[0]


def classify_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def main() -> int:
    ensure_dirs()
    labels = read_csv(MODULE4_CONTENT_LABELS)
    reply_edges = read_csv(MODULE5_REPLY_EDGES)
    hot_comments = read_csv(MODULE2_HOT_COMMENTS)

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_titles: dict[str, str] = {}
    for row in labels:
        by_source[row["source_id"]].append(row)
        source_titles[row["source_id"]] = row["source_title"]

    hot_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hot_comments:
        hot_by_source[row["source_id"]].append(row)

    reply_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reply_edges:
        reply_by_source[row["source_id"]].append(row)

    score_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    sentiment_rows: list[dict[str, Any]] = []
    stance_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    hot_stance_rows: list[dict[str, Any]] = []

    total_sources = len(by_source)
    for source_id, items in sorted(by_source.items()):
        title = source_titles.get(source_id, "")
        total = len(items)
        topic_counter = Counter(row["topic_label"] for row in items)
        sentiment_counter = Counter(row["sentiment_label"] for row in items)
        stance_counter = Counter(row["stance_label"] for row in items)
        topic_label, topic_count = dominant(topic_counter)
        sentiment_label, sentiment_count = dominant(sentiment_counter)
        stance_label, stance_count = dominant(stance_counter)
        topic_concentration = ratio(topic_count, total)
        sentiment_consistency = ratio(sentiment_count, total)
        stance_consistency = ratio(stance_count, total)

        text_score = round(
            TEXT_WEIGHTS["topic_concentration"] * topic_concentration
            + TEXT_WEIGHTS["sentiment_consistency"] * sentiment_consistency
            + TEXT_WEIGHTS["stance_consistency"] * stance_consistency,
            4,
        )

        for label, count in topic_counter.most_common():
            topic_rows.append(
                {
                    "source_id": source_id,
                    "source_title": title,
                    "topic_label": label,
                    "content_count": count,
                    "content_percent": pct(count, total),
                    "is_dominant": "true" if label == topic_label else "false",
                }
            )
        for label, count in sentiment_counter.most_common():
            sentiment_rows.append(
                {
                    "source_id": source_id,
                    "source_title": title,
                    "sentiment_label": label,
                    "content_count": count,
                    "content_percent": pct(count, total),
                    "is_dominant": "true" if label == sentiment_label else "false",
                }
            )
        for label, count in stance_counter.most_common():
            stance_rows.append(
                {
                    "source_id": source_id,
                    "source_title": title,
                    "stance_label": label,
                    "content_count": count,
                    "content_percent": pct(count, total),
                    "is_dominant": "true" if label == stance_label else "false",
                }
            )

        edges = reply_by_source.get(source_id, [])
        reply_count = len(edges)
        same_stance_edges = sum(1 for edge in edges if edge.get("same_stance") == "true")
        cross_stance_edges = sum(1 for edge in edges if edge.get("same_stance") == "false")
        same_topic_edges = sum(1 for edge in edges if edge.get("same_topic") == "true")
        same_sentiment_edges = sum(1 for edge in edges if edge.get("same_sentiment") == "true")
        interaction_homophily = ratio(same_stance_edges, reply_count) if reply_count else 0
        structure_sample_status = "sufficient" if reply_count >= MIN_REPLY_EDGES_FOR_STRUCTURE else "insufficient"
        full_score = ""
        full_level = ""
        if structure_sample_status == "sufficient":
            full_score_value = round(
                FULL_WEIGHTS["topic_concentration"] * topic_concentration
                + FULL_WEIGHTS["sentiment_consistency"] * sentiment_consistency
                + FULL_WEIGHTS["stance_consistency"] * stance_consistency
                + FULL_WEIGHTS["interaction_homophily"] * interaction_homophily,
                4,
            )
            full_score = full_score_value
            full_level = classify_level(full_score_value)

        interaction_rows.append(
            {
                "source_id": source_id,
                "source_title": title,
                "reply_count": reply_count,
                "same_stance_edges": same_stance_edges,
                "cross_stance_edges": cross_stance_edges,
                "same_stance_percent": pct(same_stance_edges, reply_count),
                "same_topic_edges": same_topic_edges,
                "same_topic_percent": pct(same_topic_edges, reply_count),
                "same_sentiment_edges": same_sentiment_edges,
                "same_sentiment_percent": pct(same_sentiment_edges, reply_count),
                "interaction_homophily": interaction_homophily,
                "structure_sample_status": structure_sample_status,
            }
        )

        top_hot = hot_by_source.get(source_id, [])[:10]
        hot_label_counter = Counter()
        hot_like_by_stance = Counter()
        content_label_map = {row["content_id"]: row for row in items}
        for hot in top_hot:
            label_row = content_label_map.get(hot["content_id"])
            if not label_row:
                continue
            stance = label_row["stance_label"]
            hot_label_counter[stance] += 1
            hot_like_by_stance[stance] += int(hot.get("like_count") or 0)
        hot_dom_stance, hot_dom_count = dominant(hot_label_counter)
        hot_stance_rows.append(
            {
                "source_id": source_id,
                "source_title": title,
                "hot_comment_count": len(top_hot),
                "dominant_hot_stance": hot_dom_stance,
                "dominant_hot_stance_count": hot_dom_count,
                "dominant_hot_stance_percent": pct(hot_dom_count, len(top_hot)),
                "hot_stance_distribution": json.dumps(dict(hot_label_counter), ensure_ascii=False),
                "hot_stance_like_distribution": json.dumps(dict(hot_like_by_stance), ensure_ascii=False),
            }
        )

        score_rows.append(
            {
                "source_id": source_id,
                "source_title": title,
                "content_count": total,
                "dominant_topic": topic_label,
                "topic_concentration": topic_concentration,
                "dominant_sentiment": sentiment_label,
                "sentiment_consistency": sentiment_consistency,
                "dominant_stance": stance_label,
                "stance_consistency": stance_consistency,
                "reply_count": reply_count,
                "interaction_homophily": interaction_homophily,
                "structure_sample_status": structure_sample_status,
                "text_cocoon_score": text_score,
                "text_cocoon_level": classify_level(text_score),
                "full_cocoon_score": full_score,
                "full_cocoon_level": full_level,
                "dominant_hot_stance": hot_dom_stance,
                "dominant_hot_stance_percent": pct(hot_dom_count, len(top_hot)),
            }
        )

    score_rows.sort(
        key=lambda row: (
            -float(row["full_cocoon_score"] if row["full_cocoon_score"] != "" else row["text_cocoon_score"]),
            -float(row["text_cocoon_score"]),
            row["source_id"],
        )
    )

    valid_full_scores = [float(row["full_cocoon_score"]) for row in score_rows if row["full_cocoon_score"] != ""]
    text_scores = [float(row["text_cocoon_score"]) for row in score_rows]
    level_counter = Counter(row["text_cocoon_level"] for row in score_rows)
    structure_counter = Counter(row["structure_sample_status"] for row in score_rows)

    overview = {
        "module": "module6_cocoon",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_count": total_sources,
        "text_score_weights": TEXT_WEIGHTS,
        "full_score_weights": FULL_WEIGHTS,
        "min_reply_edges_for_structure": MIN_REPLY_EDGES_FOR_STRUCTURE,
        "average_text_cocoon_score": round(sum(text_scores) / len(text_scores), 4) if text_scores else 0,
        "average_full_cocoon_score": round(sum(valid_full_scores) / len(valid_full_scores), 4) if valid_full_scores else "",
        "text_level_distribution": dict(level_counter),
        "structure_sample_distribution": dict(structure_counter),
        "top_sources": score_rows[:10],
        "notes": [
            "text_cocoon_score 只基于主题、情绪、立场集中度，所有视频均可计算。",
            "full_cocoon_score 在文本分数基础上加入 reply 同立场互动比例；reply 边少于阈值时标记为结构样本不足。",
            "模块 3/4 暂未人工校验，因此本模块结果用于原型分析和趋势判断。",
        ],
    }

    write_csv(
        MODULE_DIR / "source_cocoon_scores.csv",
        score_rows,
        [
            "source_id",
            "source_title",
            "content_count",
            "dominant_topic",
            "topic_concentration",
            "dominant_sentiment",
            "sentiment_consistency",
            "dominant_stance",
            "stance_consistency",
            "reply_count",
            "interaction_homophily",
            "structure_sample_status",
            "text_cocoon_score",
            "text_cocoon_level",
            "full_cocoon_score",
            "full_cocoon_level",
            "dominant_hot_stance",
            "dominant_hot_stance_percent",
        ],
    )
    write_csv(MODULE_DIR / "topic_concentration.csv", topic_rows, ["source_id", "source_title", "topic_label", "content_count", "content_percent", "is_dominant"])
    write_csv(MODULE_DIR / "sentiment_consistency.csv", sentiment_rows, ["source_id", "source_title", "sentiment_label", "content_count", "content_percent", "is_dominant"])
    write_csv(MODULE_DIR / "stance_consistency.csv", stance_rows, ["source_id", "source_title", "stance_label", "content_count", "content_percent", "is_dominant"])
    write_csv(
        MODULE_DIR / "interaction_homophily.csv",
        interaction_rows,
        [
            "source_id",
            "source_title",
            "reply_count",
            "same_stance_edges",
            "cross_stance_edges",
            "same_stance_percent",
            "same_topic_edges",
            "same_topic_percent",
            "same_sentiment_edges",
            "same_sentiment_percent",
            "interaction_homophily",
            "structure_sample_status",
        ],
    )
    write_csv(
        MODULE_DIR / "hot_comment_stance_concentration.csv",
        hot_stance_rows,
        [
            "source_id",
            "source_title",
            "hot_comment_count",
            "dominant_hot_stance",
            "dominant_hot_stance_count",
            "dominant_hot_stance_percent",
            "hot_stance_distribution",
            "hot_stance_like_distribution",
        ],
    )
    write_json(MODULE_DIR / "cocoon_overview.json", overview)

    top_lines = "\n".join(
        f"{idx}. {row['source_title']} | text={row['text_cocoon_score']}({row['text_cocoon_level']}) | full={row['full_cocoon_score'] or '样本不足'} | 主立场={row['dominant_stance']}({round(float(row['stance_consistency'])*100, 2)}%)"
        for idx, row in enumerate(score_rows[:8], start=1)
    )
    summary_md = f"""# 模块 6：评论区圈层化分析

## 指标设计

```text
text_cocoon_score =
  0.30 * 主题集中度
+ 0.30 * 情绪一致性
+ 0.40 * 立场一致性

full_cocoon_score =
  0.25 * 主题集中度
+ 0.25 * 情绪一致性
+ 0.30 * 立场一致性
+ 0.20 * 互动同质性
```

其中：

```text
主题集中度 = 视频内最大主题占比
情绪一致性 = 视频内最大情绪占比
立场一致性 = 视频内最大立场占比
互动同质性 = reply 边中同立场互动比例
```

当某视频 reply 边数少于 {MIN_REPLY_EDGES_FOR_STRUCTURE} 条时，`full_cocoon_score` 标记为结构样本不足，只保留文本版分数。

## 总体结果

```text
视频源数量：{total_sources}
平均 text_cocoon_score：{overview['average_text_cocoon_score']}
平均 full_cocoon_score：{overview['average_full_cocoon_score']}
文本圈层等级分布：{dict(level_counter)}
结构样本状态分布：{dict(structure_counter)}
```

## 圈层化较高的视频

```text
{top_lines}
```

## 解释边界

当前模块衡量的是 B 站视频评论区中的局部圈层化倾向，不代表全网舆情传播网络。由于模块 3/4 暂未人工校验，且部分视频 reply 边较少，结果适合作为课程设计原型分析和趋势判断。

## 输出文件

```text
data/analysis/module6_cocoon/cocoon_overview.json
data/analysis/module6_cocoon/source_cocoon_scores.csv
data/analysis/module6_cocoon/topic_concentration.csv
data/analysis/module6_cocoon/sentiment_consistency.csv
data/analysis/module6_cocoon/stance_consistency.csv
data/analysis/module6_cocoon/interaction_homophily.csv
data/analysis/module6_cocoon/hot_comment_stance_concentration.csv
```
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"cocoon_overview: {MODULE_DIR / 'cocoon_overview.json'}")
    print(f"source_scores: {MODULE_DIR / 'source_cocoon_scores.csv'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


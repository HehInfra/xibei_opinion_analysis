from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "xibei_event.db"
MODULE4_PATH = BASE_DIR / "analysis" / "module4_sentiment_stance" / "content_sentiment_stance.csv"
MODULE_DIR = BASE_DIR / "analysis" / "module5_network"


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, data: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def pct(n: int | float, total: int | float) -> float:
    return round(n / total * 100, 2) if total else 0.0


def dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def build_components(nodes: set[str], undirected_adj: dict[str, set[str]]) -> list[set[str]]:
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in nodes:
        if node in visited:
            continue
        comp: set[str] = set()
        queue: deque[str] = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            comp.add(current)
            for neighbor in undirected_adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def main() -> int:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    content_labels = {row["content_id"]: row for row in read_csv(MODULE4_PATH)}

    users = {
        row["user_id"]: row
        for row in rows(
            conn,
            """
            select user_id, user_name, level, first_seen_time, last_seen_time
            from users
            """,
        )
    }
    sources = {
        row["source_id"]: row
        for row in rows(
            conn,
            """
            select source_id, source_title
            from sources
            """,
        )
    }
    reply_relations = rows(
        conn,
        """
        select
          relation_id,
          platform,
          source_id,
          source_user_id,
          target_user_id,
          relation_type,
          content_id,
          target_content_id,
          created_at,
          weight
        from relations
        where relation_type = 'reply'
          and source_user_id != ''
          and target_user_id != ''
        order by created_at
        """,
    )

    user_content_counter: dict[str, Counter[str]] = defaultdict(Counter)
    user_topic_counter: dict[str, Counter[str]] = defaultdict(Counter)
    user_sentiment_counter: dict[str, Counter[str]] = defaultdict(Counter)
    user_stance_counter: dict[str, Counter[str]] = defaultdict(Counter)
    user_like_sum: Counter[str] = Counter()
    for item in content_labels.values():
        user_id = item.get("user_id")
        # content_topics from module 3 did not carry user_id; module 4 also does not.
        # Fill this below from SQLite contents table.
    content_user_rows = rows(conn, "select content_id, user_id from contents")
    content_to_user = {row["content_id"]: row["user_id"] for row in content_user_rows}
    for content_id, item in content_labels.items():
        user_id = content_to_user.get(content_id, "")
        if not user_id:
            continue
        user_content_counter[user_id]["content_count"] += 1
        user_topic_counter[user_id][item.get("topic_label", "")] += 1
        user_sentiment_counter[user_id][item.get("sentiment_label", "")] += 1
        user_stance_counter[user_id][item.get("stance_label", "")] += 1
        try:
            user_like_sum[user_id] += int(item.get("like_count") or 0)
        except ValueError:
            pass

    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    edge_weight: Counter[tuple[str, str]] = Counter()
    source_count_by_user: dict[str, set[str]] = defaultdict(set)
    stance_matrix: Counter[tuple[str, str]] = Counter()
    sentiment_matrix: Counter[tuple[str, str]] = Counter()
    topic_matrix: Counter[tuple[str, str]] = Counter()
    source_stance_matrix: Counter[tuple[str, str, str]] = Counter()
    undirected_adj: dict[str, set[str]] = defaultdict(set)
    node_ids: set[str] = set()

    edge_rows: list[dict[str, Any]] = []
    for rel in reply_relations:
        src = rel["source_user_id"]
        tgt = rel["target_user_id"]
        source_content = content_labels.get(rel["content_id"], {})
        target_content = content_labels.get(rel["target_content_id"], {})
        source_stance = source_content.get("stance_label", "")
        target_stance = target_content.get("stance_label", "") or dominant(user_stance_counter.get(tgt, Counter()))
        source_sentiment = source_content.get("sentiment_label", "")
        target_sentiment = target_content.get("sentiment_label", "") or dominant(user_sentiment_counter.get(tgt, Counter()))
        source_topic = source_content.get("topic_label", "")
        target_topic = target_content.get("topic_label", "") or dominant(user_topic_counter.get(tgt, Counter()))

        out_degree[src] += 1
        in_degree[tgt] += 1
        edge_weight[(src, tgt)] += 1
        source_count_by_user[src].add(rel["source_id"])
        source_count_by_user[tgt].add(rel["source_id"])
        stance_matrix[(source_stance, target_stance)] += 1
        sentiment_matrix[(source_sentiment, target_sentiment)] += 1
        topic_matrix[(source_topic, target_topic)] += 1
        source_stance_matrix[(rel["source_id"], source_stance, target_stance)] += 1
        undirected_adj[src].add(tgt)
        undirected_adj[tgt].add(src)
        node_ids.add(src)
        node_ids.add(tgt)

        edge_rows.append(
            {
                "relation_id": rel["relation_id"],
                "source_id": rel["source_id"],
                "source_title": sources.get(rel["source_id"], {}).get("source_title", ""),
                "source_user_id": src,
                "source_user_name": users.get(src, {}).get("user_name", ""),
                "target_user_id": tgt,
                "target_user_name": users.get(tgt, {}).get("user_name", ""),
                "content_id": rel["content_id"],
                "target_content_id": rel["target_content_id"],
                "created_at": rel["created_at"],
                "source_topic": source_topic,
                "target_topic": target_topic,
                "source_sentiment": source_sentiment,
                "target_sentiment": target_sentiment,
                "source_stance": source_stance,
                "target_stance": target_stance,
                "same_topic": "true" if source_topic and source_topic == target_topic else "false",
                "same_sentiment": "true" if source_sentiment and source_sentiment == target_sentiment else "false",
                "same_stance": "true" if source_stance and source_stance == target_stance else "false",
                "weight": rel["weight"],
            }
        )

    components = build_components(node_ids, undirected_adj)
    component_index: dict[str, int] = {}
    for idx, comp in enumerate(components, start=1):
        for node in comp:
            component_index[node] = idx

    node_rows: list[dict[str, Any]] = []
    for user_id in sorted(node_ids):
        indeg = in_degree[user_id]
        outdeg = out_degree[user_id]
        node_rows.append(
            {
                "user_id": user_id,
                "user_name": users.get(user_id, {}).get("user_name", ""),
                "in_degree": indeg,
                "out_degree": outdeg,
                "total_degree": indeg + outdeg,
                "source_count": len(source_count_by_user[user_id]),
                "content_count": user_content_counter[user_id]["content_count"],
                "received_like_sum": user_like_sum[user_id],
                "dominant_topic": dominant(user_topic_counter[user_id]),
                "dominant_sentiment": dominant(user_sentiment_counter[user_id]),
                "dominant_stance": dominant(user_stance_counter[user_id]),
                "component_id": component_index.get(user_id, ""),
            }
        )
    node_rows.sort(key=lambda row: (-int(row["total_degree"]), -int(row["in_degree"]), row["user_id"]))

    key_users: list[dict[str, Any]] = []
    for rank, row in enumerate(node_rows[:50], start=1):
        key_users.append({"rank": rank, **row})

    aggregated_edges = [
        {
            "source_user_id": src,
            "source_user_name": users.get(src, {}).get("user_name", ""),
            "target_user_id": tgt,
            "target_user_name": users.get(tgt, {}).get("user_name", ""),
            "weight": weight,
            "source_dominant_stance": dominant(user_stance_counter[src]),
            "target_dominant_stance": dominant(user_stance_counter[tgt]),
            "same_dominant_stance": "true" if dominant(user_stance_counter[src]) == dominant(user_stance_counter[tgt]) else "false",
        }
        for (src, tgt), weight in edge_weight.items()
    ]
    aggregated_edges.sort(key=lambda row: (-int(row["weight"]), row["source_user_id"], row["target_user_id"]))

    stance_matrix_rows = [
        {
            "source_stance": src,
            "target_stance": tgt,
            "reply_count": count,
            "same_stance": "true" if src and src == tgt else "false",
        }
        for (src, tgt), count in sorted(stance_matrix.items(), key=lambda item: (-item[1], item[0]))
    ]
    sentiment_matrix_rows = [
        {
            "source_sentiment": src,
            "target_sentiment": tgt,
            "reply_count": count,
            "same_sentiment": "true" if src and src == tgt else "false",
        }
        for (src, tgt), count in sorted(sentiment_matrix.items(), key=lambda item: (-item[1], item[0]))
    ]
    topic_matrix_rows = [
        {
            "source_topic": src,
            "target_topic": tgt,
            "reply_count": count,
            "same_topic": "true" if src and src == tgt else "false",
        }
        for (src, tgt), count in sorted(topic_matrix.items(), key=lambda item: (-item[1], item[0]))
    ]
    source_stance_rows = [
        {
            "source_id": source_id,
            "source_title": sources.get(source_id, {}).get("source_title", ""),
            "source_stance": src_stance,
            "target_stance": tgt_stance,
            "reply_count": count,
            "same_stance": "true" if src_stance and src_stance == tgt_stance else "false",
        }
        for (source_id, src_stance, tgt_stance), count in sorted(source_stance_matrix.items(), key=lambda item: (item[0][0], -item[1]))
    ]

    total_reply_edges = len(edge_rows)
    same_stance_count = sum(1 for row in edge_rows if row["same_stance"] == "true")
    cross_stance_count = sum(1 for row in edge_rows if row["same_stance"] == "false" and row["source_stance"] and row["target_stance"])
    same_topic_count = sum(1 for row in edge_rows if row["same_topic"] == "true")
    same_sentiment_count = sum(1 for row in edge_rows if row["same_sentiment"] == "true")
    node_count = len(node_ids)
    unique_edge_count = len(edge_weight)
    density = round(unique_edge_count / (node_count * (node_count - 1)), 6) if node_count > 1 else 0

    graph_json = {
        "directed": True,
        "nodes": [
            {
                "id": row["user_id"],
                "name": row["user_name"],
                "in_degree": row["in_degree"],
                "out_degree": row["out_degree"],
                "total_degree": row["total_degree"],
                "dominant_topic": row["dominant_topic"],
                "dominant_sentiment": row["dominant_sentiment"],
                "dominant_stance": row["dominant_stance"],
                "component_id": row["component_id"],
            }
            for row in node_rows
        ],
        "links": [
            {
                "source": row["source_user_id"],
                "target": row["target_user_id"],
                "weight": row["weight"],
                "same_dominant_stance": row["same_dominant_stance"],
            }
            for row in aggregated_edges
        ],
    }

    overview = {
        "module": "module5_network",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "network_type": "reply_strong_relation",
        "node_count": node_count,
        "reply_edge_count": total_reply_edges,
        "unique_directed_edge_count": unique_edge_count,
        "density": density,
        "component_count": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "same_stance_reply_count": same_stance_count,
        "cross_stance_reply_count": cross_stance_count,
        "same_stance_reply_percent": pct(same_stance_count, total_reply_edges),
        "cross_stance_reply_percent": pct(cross_stance_count, total_reply_edges),
        "same_topic_reply_count": same_topic_count,
        "same_topic_reply_percent": pct(same_topic_count, total_reply_edges),
        "same_sentiment_reply_count": same_sentiment_count,
        "same_sentiment_reply_percent": pct(same_sentiment_count, total_reply_edges),
        "top_key_users": key_users[:10],
        "notes": [
            "本模块优先分析真实楼中楼 reply 强关系。",
            "comment_source 是用户评论视频的二部关系，未纳入用户对用户强关系网络。",
            "co_participation 共同评论同视频属于可构造弱关系，本版暂不作为主网络结论。",
            "模块 4 人工校验暂未执行，因此立场同质性结果用于原型分析和趋势判断。",
        ],
    }

    write_csv(
        MODULE_DIR / "reply_edges.csv",
        edge_rows,
        [
            "relation_id",
            "source_id",
            "source_title",
            "source_user_id",
            "source_user_name",
            "target_user_id",
            "target_user_name",
            "content_id",
            "target_content_id",
            "created_at",
            "source_topic",
            "target_topic",
            "source_sentiment",
            "target_sentiment",
            "source_stance",
            "target_stance",
            "same_topic",
            "same_sentiment",
            "same_stance",
            "weight",
        ],
    )
    write_csv(
        MODULE_DIR / "user_nodes.csv",
        node_rows,
        [
            "user_id",
            "user_name",
            "in_degree",
            "out_degree",
            "total_degree",
            "source_count",
            "content_count",
            "received_like_sum",
            "dominant_topic",
            "dominant_sentiment",
            "dominant_stance",
            "component_id",
        ],
    )
    write_csv(MODULE_DIR / "key_users.csv", key_users, ["rank"] + list(node_rows[0].keys()) if node_rows else ["rank"])
    write_csv(
        MODULE_DIR / "aggregated_reply_edges.csv",
        aggregated_edges,
        [
            "source_user_id",
            "source_user_name",
            "target_user_id",
            "target_user_name",
            "weight",
            "source_dominant_stance",
            "target_dominant_stance",
            "same_dominant_stance",
        ],
    )
    write_csv(MODULE_DIR / "stance_interaction_matrix.csv", stance_matrix_rows, ["source_stance", "target_stance", "reply_count", "same_stance"])
    write_csv(MODULE_DIR / "sentiment_interaction_matrix.csv", sentiment_matrix_rows, ["source_sentiment", "target_sentiment", "reply_count", "same_sentiment"])
    write_csv(MODULE_DIR / "topic_interaction_matrix.csv", topic_matrix_rows, ["source_topic", "target_topic", "reply_count", "same_topic"])
    write_csv(MODULE_DIR / "source_stance_interaction_matrix.csv", source_stance_rows, ["source_id", "source_title", "source_stance", "target_stance", "reply_count", "same_stance"])
    write_json(MODULE_DIR / "network_graph.json", graph_json)
    write_json(MODULE_DIR / "network_overview.json", overview)

    top_user_lines = "\n".join(
        f"{row['rank']}. {row['user_name']} | 入度 {row['in_degree']} | 出度 {row['out_degree']} | 主立场 {row['dominant_stance']}"
        for row in key_users[:10]
    )
    summary_md = f"""# 模块 5：用户互动网络分析

## 分析对象

```text
网络类型：楼中楼 reply 强关系网络
节点：参与 reply 关系的 B站用户
边：用户 A 回复 用户 B
```

## 核心统计

```text
节点数：{node_count}
reply 边数：{total_reply_edges}
去重有向边数：{unique_edge_count}
网络密度：{density}
弱连通分量数：{len(components)}
最大连通分量节点数：{len(components[0]) if components else 0}
```

## 标签同质性

```text
同立场回复数：{same_stance_count}
同立场回复占比：{pct(same_stance_count, total_reply_edges)}%
跨立场回复数：{cross_stance_count}
跨立场回复占比：{pct(cross_stance_count, total_reply_edges)}%
同主题回复占比：{pct(same_topic_count, total_reply_edges)}%
同情绪回复占比：{pct(same_sentiment_count, total_reply_edges)}%
```

## 关键用户 Top 10

```text
{top_user_lines}
```

## 方法说明

当前模块只将真实 `reply` 关系纳入强关系网络。`comment_source` 表示用户评论视频，不是用户对用户互动，因此不进入主网络。共同评论同一视频的 `co_participation` 弱关系可在后续扩展，但本版不混入强关系结论。

## 输出文件

```text
data/analysis/module5_network/network_overview.json
data/analysis/module5_network/reply_edges.csv
data/analysis/module5_network/user_nodes.csv
data/analysis/module5_network/key_users.csv
data/analysis/module5_network/stance_interaction_matrix.csv
data/analysis/module5_network/network_graph.json
```
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"network_overview: {MODULE_DIR / 'network_overview.json'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    print(f"reply_edges: {MODULE_DIR / 'reply_edges.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

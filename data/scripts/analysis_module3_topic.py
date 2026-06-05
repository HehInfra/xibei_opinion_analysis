from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepseek_config import load_deepseek_config

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "xibei_event.db"
MODULE_DIR = BASE_DIR / "analysis" / "module3_topic"
LLM_RAW_DIR = MODULE_DIR / "llm_raw"
DEFAULT_DEEPSEEK_CONFIG = BASE_DIR / "config" / "deepseek.env"


LABELS = [
    "prepared_food_authenticity",
    "price_value",
    "food_safety_child_meal",
    "brand_pr_crisis",
    "luoyonghao_persona",
    "consumer_right_to_know",
    "catering_industry_general",
    "mockery_meme",
    "other_unclear",
]


TOPIC_DEFS = {
    "prepared_food_authenticity": "预制菜、现做、冷冻、中央厨房、食材真实性、新鲜程度、现宰等。",
    "price_value": "价格、贵不贵、性价比、人均、馒头价格、与其他餐厅价格比较。",
    "food_safety_child_meal": "食品安全、儿童餐、孩子、健康、转基因油、添加剂、过期等。",
    "brand_pr_crisis": "品牌公关、回应、声明、自证、翻车、自爆、贾国龙、开放后厨、道歉等。",
    "luoyonghao_persona": "罗永浩本人、老罗、较真、维权经历、直播表现、西门子、锤子等。",
    "consumer_right_to_know": "消费者知情权、透明、告知、隐瞒、选择权、真实信息。",
    "catering_industry_general": "餐饮行业共性、外卖、连锁餐饮、行业普遍做法、餐厅饭店比较。",
    "mockery_meme": "主要是调侃、玩梗、讽刺、哈哈、笑死、doge、夸张表达，实质信息较少。",
    "other_unclear": "无法明确归类、文本过短、上下文不足或不属于以上主题。",
}


RULES = {
    "food_safety_child_meal": ["儿童餐", "孩子", "小孩", "宝宝", "健康", "转基因", "大豆油", "添加剂", "食品安全", "过期", "不敢吃"],
    "consumer_right_to_know": ["知情权", "消费者", "告知", "透明", "隐瞒", "明示", "选择权", "真实", "真诚"],
    "brand_pr_crisis": ["公关", "回应", "声明", "自证", "自爆", "翻车", "贾国龙", "开放后厨", "危机公关", "道歉", "捅了自己", "自杀式"],
    "prepared_food_authenticity": ["预制菜", "现做", "现炒", "冷冻", "冻货", "中央厨房", "保质期", "隔夜", "加热", "新鲜", "食材", "现宰", "预制", "复热"],
    "price_value": ["贵", "价格", "性价比", "人均", "21块", "21元", "馒头", "黑珍珠", "米其林", "萨莉亚", "麦当劳", "肯德基", "包子", "便宜"],
    "luoyonghao_persona": ["罗永浩", "老罗", "较真", "维权", "西门子", "直播", "锤子", "王自如", "怼", "消费者维权"],
    "catering_industry_general": ["餐饮", "行业", "外卖", "连锁", "后厨", "餐厅", "饭店", "麦当劳", "肯德基", "萨莉亚", "必胜客"],
    "mockery_meme": ["哈哈", "笑死", "绷不住", "doge", "大哭", "乐", "梗", "离谱", "现宰", "捅自己", "抽象", "蚌埠住"],
}


PRIORITY = [
    "food_safety_child_meal",
    "consumer_right_to_know",
    "brand_pr_crisis",
    "prepared_food_authenticity",
    "price_value",
    "luoyonghao_persona",
    "catering_industry_general",
    "mockery_meme",
    "other_unclear",
]


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def compact_text(text: str, limit: int = 700) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def keyword_hits(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for label, keywords in RULES.items():
        hits = [kw for kw in keywords if kw.lower() in text.lower()]
        if hits:
            result[label] = hits
    return result


def rule_classify(text: str) -> dict[str, Any]:
    hits = keyword_hits(text)
    if not hits:
        return {
            "topic_label": "other_unclear",
            "topic_confidence": 0.40,
            "topic_reason": "未命中明确主题关键词，暂归为其他/不明确。",
            "topic_matched_keywords": "",
            "topic_candidate_labels": "",
            "need_review": "true",
        }

    candidate_labels = [label for label in PRIORITY if label in hits]
    selected = candidate_labels[0]
    hit_count = sum(len(v) for v in hits.values())
    selected_hits = hits[selected]
    competing = len(candidate_labels) > 1
    if len(selected_hits) >= 2 or hit_count >= 3:
        confidence = 0.90 if not competing else 0.75
    elif len(selected_hits) == 1:
        confidence = 0.75 if not competing else 0.60
    else:
        confidence = 0.60
    return {
        "topic_label": selected,
        "topic_confidence": confidence,
        "topic_reason": f"规则命中 {selected} 关键词：{', '.join(selected_hits)}。",
        "topic_matched_keywords": json.dumps(hits, ensure_ascii=False),
        "topic_candidate_labels": ",".join(candidate_labels),
        "need_review": "true" if confidence < 0.70 or competing or selected == "other_unclear" else "false",
    }


def extract_json_object(text: str) -> Any:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start >= 0 and array_end > array_start and (object_start < 0 or array_start < object_start):
        return json.loads(cleaned[array_start : array_end + 1])
    if object_start >= 0 and object_end > object_start:
        return json.loads(cleaned[object_start : object_end + 1])
    raise


def normalize_llm_items(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        items = parsed.get("items", [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def call_deepseek(
    *,
    api_key: str,
    endpoint: str,
    model: str,
    batch: list[dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    system = (
        "你是中文社交媒体舆情文本主题分类助手。"
        "只能从给定标签中选择一个主主题。"
        "请严格输出 JSON，不要输出 Markdown。"
    )
    label_lines = "\n".join(f"- {label}: {desc}" for label, desc in TOPIC_DEFS.items())
    user = {
        "task": "对每条 B站评论进行西贝事件主题分类。",
        "allowed_labels": LABELS,
        "label_definitions": label_lines,
        "rules": [
            "每条评论只选择一个最主要 topic_label。",
            "如果评论是调侃但明确讨论价格、预制菜、公关等实质主题，优先选择实质主题。",
            "只有当主要价值是玩梗且缺少实质主题时，选择 mockery_meme。",
            "无法判断时选择 other_unclear。",
            "confidence 取 0.40 到 0.95。",
        ],
        "output_schema": {
            "items": [
                {
                    "content_id": "string",
                    "topic_label": "one of allowed_labels",
                    "confidence": "number",
                    "reason": "short Chinese reason",
                }
            ]
        },
        "items": batch,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": getattr(resp, "status", None), "body": json.loads(response_body)}
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": str(exc), "body_text": response_body}
    except (URLError, TimeoutError) as exc:
        return {"ok": False, "status": None, "error": str(exc), "body_text": ""}


def build_content_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return rows(
        conn,
        """
        select
          c.content_id,
          c.source_id,
          s.source_title,
          c.content_text,
          c.content_type,
          c.created_at,
          cast(nullif(c.like_count, '') as integer) as like_count
        from contents c
        left join sources s on s.source_id = c.source_id
        order by c.content_id
        """,
    )


def apply_llm_labels(
    labeled_rows: list[dict[str, Any]],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    batch_size: int,
    timeout: int,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["content_id"]: row for row in labeled_rows}
    logs: list[dict[str, Any]] = []
    items = [
        {
            "content_id": row["content_id"],
            "source_title": row["source_title"],
            "content_text": compact_text(row["content_text"]),
            "rule_topic_label": row["rule_topic_label"],
            "rule_reason": row["rule_topic_reason"],
        }
        for row in labeled_rows
    ]

    for start in range(0, len(items), batch_size):
        batch_no = start // batch_size + 1
        batch = items[start : start + batch_size]
        result = call_deepseek(api_key=api_key, endpoint=endpoint, model=model, batch=batch, timeout=timeout)
        raw_path = LLM_RAW_DIR / f"batch_{batch_no:03d}.json"
        write_json(raw_path, result)
        log = {
            "batch_no": batch_no,
            "item_count": len(batch),
            "ok": result.get("ok"),
            "status": result.get("status"),
            "raw_path": str(raw_path),
            "error": result.get("error", ""),
        }
        if not result.get("ok"):
            logs.append(log)
            continue
        try:
            content = result["body"]["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            llm_items = normalize_llm_items(parsed)
            applied = 0
            for item in llm_items:
                content_id = item.get("content_id")
                label = item.get("topic_label")
                if content_id not in by_id or label not in LABELS:
                    continue
                confidence = float(item.get("confidence", 0.75))
                by_id[content_id]["llm_topic_label"] = label
                by_id[content_id]["llm_topic_confidence"] = round(max(0.4, min(confidence, 0.95)), 2)
                by_id[content_id]["llm_topic_reason"] = str(item.get("reason", "")).strip()
                by_id[content_id]["topic_label"] = label
                by_id[content_id]["topic_confidence"] = by_id[content_id]["llm_topic_confidence"]
                by_id[content_id]["topic_reason"] = "LLM校正：" + by_id[content_id]["llm_topic_reason"]
                by_id[content_id]["label_method"] = "rule+llm"
                if label == "other_unclear" or by_id[content_id]["topic_confidence"] < 0.70:
                    by_id[content_id]["need_review"] = "true"
                applied += 1
            log["applied"] = applied
        except Exception as exc:  # noqa: BLE001
            log["error"] = f"parse_or_apply_error: {exc}"
        logs.append(log)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return labeled_rows, logs


def apply_saved_llm_raw(labeled_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["content_id"]: row for row in labeled_rows}
    logs: list[dict[str, Any]] = []
    for raw_path in sorted(LLM_RAW_DIR.glob("batch_*.json")):
        batch_no_match = re.search(r"batch_(\d+)", raw_path.name)
        batch_no = int(batch_no_match.group(1)) if batch_no_match else 0
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        log = {
            "batch_no": batch_no,
            "item_count": "",
            "ok": result.get("ok"),
            "status": result.get("status"),
            "raw_path": str(raw_path),
            "error": "",
        }
        if not result.get("ok"):
            log["error"] = result.get("error", "")
            logs.append(log)
            continue
        try:
            content = result["body"]["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            llm_items = normalize_llm_items(parsed)
            applied = 0
            for item in llm_items:
                content_id = item.get("content_id")
                label = item.get("topic_label")
                if content_id not in by_id or label not in LABELS:
                    continue
                confidence = float(item.get("confidence", 0.75))
                by_id[content_id]["llm_topic_label"] = label
                by_id[content_id]["llm_topic_confidence"] = round(max(0.4, min(confidence, 0.95)), 2)
                by_id[content_id]["llm_topic_reason"] = str(item.get("reason", "")).strip()
                by_id[content_id]["topic_label"] = label
                by_id[content_id]["topic_confidence"] = by_id[content_id]["llm_topic_confidence"]
                by_id[content_id]["topic_reason"] = "LLM校正：" + by_id[content_id]["llm_topic_reason"]
                by_id[content_id]["label_method"] = "rule+llm"
                if label == "other_unclear" or by_id[content_id]["topic_confidence"] < 0.70:
                    by_id[content_id]["need_review"] = "true"
                applied += 1
            log["item_count"] = len(llm_items)
            log["applied"] = applied
        except Exception as exc:  # noqa: BLE001
            log["error"] = f"parse_or_apply_error: {exc}"
        logs.append(log)
    return labeled_rows, logs


def summarize_by_label(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(row["topic_label"] for row in rows_)
    like_counter: Counter[str] = Counter()
    for row in rows_:
        like_counter[row["topic_label"]] += int(row.get("like_count") or 0)
    total = len(rows_)
    return {
        label: {
            "content_count": counter[label],
            "content_percent": round(counter[label] / total * 100, 2) if total else 0,
            "like_sum": like_counter[label],
        }
        for label in LABELS
        if counter[label] > 0
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Module 3 topic analysis: rules + optional DeepSeek.")
    parser.add_argument("--use-llm", action="store_true", help="Call DeepSeek API to refine rule labels.")
    parser.add_argument("--reuse-llm-raw", action="store_true", help="Reuse saved llm_raw/batch_*.json without calling API.")
    parser.add_argument("--config", default=str(DEFAULT_DEEPSEEK_CONFIG), help="DeepSeek env config file path.")
    parser.add_argument("--endpoint", default=None, help="Override DeepSeek endpoint from config/env.")
    parser.add_argument("--model", default=None, help="Override DeepSeek model from config/env.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()
    deepseek_config = load_deepseek_config(args.config)
    endpoint = args.endpoint or deepseek_config.endpoint
    model = args.model or deepseek_config.model

    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    content_rows = build_content_rows(conn)
    labeled_rows: list[dict[str, Any]] = []

    for row in content_rows:
        rule = rule_classify(row["content_text"] or "")
        labeled_rows.append(
            {
                "content_id": row["content_id"],
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                "content_type": row["content_type"],
                "content_text": row["content_text"],
                "created_at": row["created_at"],
                "like_count": row.get("like_count") or 0,
                "rule_topic_label": rule["topic_label"],
                "rule_topic_confidence": rule["topic_confidence"],
                "rule_topic_reason": rule["topic_reason"],
                "topic_label": rule["topic_label"],
                "topic_confidence": rule["topic_confidence"],
                "topic_reason": rule["topic_reason"],
                "topic_matched_keywords": rule["topic_matched_keywords"],
                "topic_candidate_labels": rule["topic_candidate_labels"],
                "llm_topic_label": "",
                "llm_topic_confidence": "",
                "llm_topic_reason": "",
                "need_review": rule["need_review"],
                "label_method": "rule",
            }
        )

    llm_logs: list[dict[str, Any]] = []
    if args.reuse_llm_raw:
        labeled_rows, llm_logs = apply_saved_llm_raw(labeled_rows)
    elif args.use_llm:
        api_key = deepseek_config.api_key
        if not api_key:
            raise SystemExit(
                f"Missing DEEPSEEK_API_KEY. Set it in environment or copy data/config/deepseek.example.env to {args.config}."
            )
        labeled_rows, llm_logs = apply_llm_labels(
            labeled_rows,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            batch_size=args.batch_size,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )

    fieldnames = [
        "content_id",
        "source_id",
        "source_title",
        "content_type",
        "content_text",
        "created_at",
        "like_count",
        "rule_topic_label",
        "rule_topic_confidence",
        "rule_topic_reason",
        "topic_label",
        "topic_confidence",
        "topic_reason",
        "topic_matched_keywords",
        "topic_candidate_labels",
        "llm_topic_label",
        "llm_topic_confidence",
        "llm_topic_reason",
        "need_review",
        "label_method",
    ]
    write_csv(MODULE_DIR / "content_topics.csv", labeled_rows, fieldnames)
    write_csv(
        MODULE_DIR / "llm_batches.csv",
        llm_logs,
        ["batch_no", "item_count", "ok", "status", "applied", "raw_path", "error"],
    )

    by_source: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_totals: Counter[str] = Counter()
    for row in labeled_rows:
        source_totals[row["source_id"]] += 1
    for row in labeled_rows:
        key = (row["source_id"], row["source_title"], row["topic_label"])
        if key not in by_source:
            by_source[key] = {
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                "topic_label": row["topic_label"],
                "content_count": 0,
                "content_percent": 0,
                "like_sum": 0,
                "avg_like": 0,
                "max_like": 0,
            }
        item = by_source[key]
        like = int(row.get("like_count") or 0)
        item["content_count"] += 1
        item["like_sum"] += like
        item["max_like"] = max(item["max_like"], like)
    source_summary = []
    for item in by_source.values():
        item["content_percent"] = round(item["content_count"] / source_totals[item["source_id"]] * 100, 2)
        item["avg_like"] = round(item["like_sum"] / item["content_count"], 2) if item["content_count"] else 0
        source_summary.append(item)
    source_summary.sort(key=lambda x: (x["source_id"], -x["content_count"], x["topic_label"]))
    write_csv(
        MODULE_DIR / "topic_summary_by_source.csv",
        source_summary,
        ["source_id", "source_title", "topic_label", "content_count", "content_percent", "like_sum", "avg_like", "max_like"],
    )

    overview = {
        "module": "module3_topic",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_database": str(DB_PATH),
        "use_llm": args.use_llm or args.reuse_llm_raw,
        "reused_llm_raw": args.reuse_llm_raw,
        "model": args.model if (args.use_llm or args.reuse_llm_raw) else "",
        "endpoint": args.endpoint if args.use_llm else "",
        "content_count": len(labeled_rows),
        "topic_summary": summarize_by_label(labeled_rows),
        "rule_topic_summary": summarize_by_label([{**row, "topic_label": row["rule_topic_label"]} for row in labeled_rows]),
        "need_review_count": sum(1 for row in labeled_rows if row["need_review"] == "true"),
        "llm_batches": llm_logs,
        "notes": [
            "当前版本暂跳过人工校验。",
            "最终 topic_label 优先采用 LLM 校正结果；如果 LLM 未返回该条，则保留规则初标。",
            "content_topics.csv 保留 rule 与 llm 两套字段，便于回溯比较。",
        ],
    }
    write_json(MODULE_DIR / "topic_overview.json", overview)

    summary = summarize_by_label(labeled_rows)
    top_lines = "\n".join(
        f"{label}: {data['content_count']} 条，占比 {data['content_percent']}%，获赞 {data['like_sum']}"
        for label, data in sorted(summary.items(), key=lambda item: item[1]["content_count"], reverse=True)
    )
    llm_ok = sum(1 for log in llm_logs if log.get("ok"))
    llm_applied = sum(int(log.get("applied") or 0) for log in llm_logs)
    summary_md = f"""# 模块 3：文本主题分析

## 分析方法

 ```text
 规则初标：已执行
大模型校正：{"已执行" if (args.use_llm or args.reuse_llm_raw) else "未执行"}
LLM 响应来源：{"复用已保存响应" if args.reuse_llm_raw else ("API 实时调用" if args.use_llm else "无")}
人工校验：暂时跳过
```

## 大模型调用

 ```text
模型：{args.model if (args.use_llm or args.reuse_llm_raw) else ""}
成功批次数：{llm_ok}
LLM 应用标签数：{llm_applied}
```

## 主题分布

```text
{top_lines}
```

## 质量提示

```text
需要复核数量：{sum(1 for row in labeled_rows if row["need_review"] == "true")}
```

当前结果保存在：

```text
data/analysis/module3_topic/content_topics.csv
data/analysis/module3_topic/topic_summary_by_source.csv
data/analysis/module3_topic/topic_overview.json
```
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"content_topics: {MODULE_DIR / 'content_topics.csv'}")
    print(f"topic_overview: {MODULE_DIR / 'topic_overview.json'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

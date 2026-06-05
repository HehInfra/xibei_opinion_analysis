from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepseek_config import load_deepseek_config

BASE_DIR = Path(__file__).resolve().parents[1]
MODULE3_PATH = BASE_DIR / "analysis" / "module3_topic" / "content_topics.csv"
MODULE_DIR = BASE_DIR / "analysis" / "module4_sentiment_stance"
LLM_RAW_DIR = MODULE_DIR / "llm_raw"
DEFAULT_DEEPSEEK_CONFIG = BASE_DIR / "config" / "deepseek.env"


SENTIMENT_LABELS = ["positive", "negative", "neutral", "sarcastic"]
STANCE_LABELS = [
    "question_xibei",
    "support_xibei",
    "support_luoyonghao",
    "question_luoyonghao",
    "neutral_discussion",
    "unclear_meme",
]


SENTIMENT_DEFS = {
    "positive": "认可、支持、赞赏、积极评价。",
    "negative": "不满、愤怒、质疑、失望、批评。",
    "neutral": "客观陈述、补充信息、事实讨论，情绪色彩弱。",
    "sarcastic": "调侃、反讽、夸张、玩梗，通常带嘲讽倾向。",
}


STANCE_DEFS = {
    "question_xibei": "质疑西贝价格、预制菜、食材真实性、公关回应、消费者权益等。",
    "support_xibei": "替西贝解释，认为西贝被误解，认为预制菜合理或争议被夸大。",
    "support_luoyonghao": "认可罗永浩较真、维权、揭露问题、推动透明。",
    "question_luoyonghao": "质疑罗永浩动机、表达方式、蹭流量或过度攻击。",
    "neutral_discussion": "事实补充、行业讨论、价格比较或信息说明，没有明显站队。",
    "unclear_meme": "文本过短、主要玩梗、上下文缺失，无法可靠判断立场。",
}


SENTIMENT_RULES = {
    "sarcastic": ["哈哈", "笑死", "绷不住", "doge", "大哭", "现宰", "捅自己", "抽象", "乐", "蚌埠住", "离谱"],
    "negative": ["缺德", "恶心", "失望", "太贵", "翻车", "自爆", "割韭菜", "不敢吃", "坑", "贵", "糊弄", "便宜油", "离谱"],
    "positive": ["支持", "认可", "真诚", "勇敢", "需要这样的人", "说得对", "有道理", "较真", "为消费者"],
    "neutral": ["其实", "按照", "一般来说", "行业", "说明", "解释", "不一定", "定义"],
}


STANCE_RULES = {
    "question_xibei": ["西贝太贵", "西贝", "预制菜", "自爆", "翻车", "公关", "不真诚", "知情权", "现宰", "冻货", "转基因", "大豆油", "21块", "便宜油"],
    "support_xibei": ["西贝没错", "可以理解", "行业都这样", "没必要骂", "预制菜不等于不安全", "不等于不安全", "没问题"],
    "support_luoyonghao": ["支持老罗", "支持罗永浩", "老罗说得对", "罗永浩说得对", "需要较真的人", "为消费者发声", "维权", "较真的人"],
    "question_luoyonghao": ["蹭流量", "老罗也", "炒作", "又来了", "过度", "罗永浩也不"],
    "neutral_discussion": ["行业", "餐饮", "麦当劳", "肯德基", "萨莉亚", "定义", "其实", "不一定", "一般"],
    "unclear_meme": ["哈哈", "笑死", "doge", "大哭", "绷不住"],
}


SENTIMENT_PRIORITY = ["sarcastic", "negative", "positive", "neutral"]
STANCE_PRIORITY = ["question_xibei", "support_luoyonghao", "support_xibei", "question_luoyonghao", "neutral_discussion", "unclear_meme"]


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def compact_text(text: str, limit: int = 700) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def shorten(text: str, limit: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def keyword_hits(text: str, rules: dict[str, list[str]]) -> dict[str, list[str]]:
    lower = text.lower()
    result: dict[str, list[str]] = {}
    for label, keywords in rules.items():
        hits = [kw for kw in keywords if kw.lower() in lower]
        if hits:
            result[label] = hits
    return result


def classify_by_rules(text: str, rules: dict[str, list[str]], priority: list[str], default_label: str) -> dict[str, Any]:
    hits = keyword_hits(text, rules)
    if not hits:
        return {
            "label": default_label,
            "confidence": 0.40,
            "reason": "未命中明确线索。",
            "matched_keywords": "",
            "candidate_labels": "",
            "need_review": True,
        }
    candidates = [label for label in priority if label in hits]
    selected = candidates[0]
    hit_count = sum(len(v) for v in hits.values())
    competing = len(candidates) > 1
    if len(hits[selected]) >= 2 or hit_count >= 3:
        confidence = 0.90 if not competing else 0.75
    else:
        confidence = 0.75 if not competing else 0.60
    return {
        "label": selected,
        "confidence": confidence,
        "reason": f"规则命中 {selected} 关键词：{', '.join(hits[selected])}。",
        "matched_keywords": json.dumps(hits, ensure_ascii=False),
        "candidate_labels": ",".join(candidates),
        "need_review": confidence < 0.70 or competing,
    }


def rule_classify(row: dict[str, str]) -> dict[str, Any]:
    text = row.get("content_text", "")
    sentiment = classify_by_rules(text, SENTIMENT_RULES, SENTIMENT_PRIORITY, "neutral")
    stance = classify_by_rules(text, STANCE_RULES, STANCE_PRIORITY, "unclear_meme")
    need_review = (
        sentiment["need_review"]
        or stance["need_review"]
        or sentiment["label"] == "sarcastic"
        or stance["label"] == "unclear_meme"
        or float(row.get("like_count") or 0) >= 1000
    )
    return {
        "sentiment_label": sentiment["label"],
        "sentiment_confidence": sentiment["confidence"],
        "sentiment_reason": sentiment["reason"],
        "stance_label": stance["label"],
        "stance_confidence": stance["confidence"],
        "stance_reason": stance["reason"],
        "matched_keywords": json.dumps(
            {
                "sentiment": json.loads(sentiment["matched_keywords"]) if sentiment["matched_keywords"] else {},
                "stance": json.loads(stance["matched_keywords"]) if stance["matched_keywords"] else {},
            },
            ensure_ascii=False,
        ),
        "sentiment_candidate_labels": sentiment["candidate_labels"],
        "stance_candidate_labels": stance["candidate_labels"],
        "need_review": "true" if need_review else "false",
    }


def extract_json_payload(text: str) -> Any:
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


def call_deepseek(api_key: str, endpoint: str, model: str, batch: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    system = (
        "你是中文社交媒体舆情分析助手。"
        "任务是为西贝事件 B站评论判断主情绪和主立场。"
        "只能从给定标签中选择，严格输出 JSON。"
    )
    payload_user = {
        "task": "对每条评论进行情绪与立场分类。",
        "sentiment_labels": SENTIMENT_LABELS,
        "sentiment_definitions": SENTIMENT_DEFS,
        "stance_labels": STANCE_LABELS,
        "stance_definitions": STANCE_DEFS,
        "rules": [
            "每条评论只选择一个 sentiment_label 和一个 stance_label。",
            "情绪和立场分开判断：讽刺西贝可 sentiment=sarcastic, stance=question_xibei。",
            "如果只是客观讨论行业或事实，stance=neutral_discussion。",
            "如果主要玩梗且无法判断立场，stance=unclear_meme。",
            "confidence 取 0.40 到 0.95。",
        ],
        "output_schema": {
            "items": [
                {
                    "content_id": "string",
                    "sentiment_label": "one of sentiment_labels",
                    "sentiment_confidence": "number",
                    "sentiment_reason": "short Chinese reason",
                    "stance_label": "one of stance_labels",
                    "stance_confidence": "number",
                    "stance_reason": "short Chinese reason",
                }
            ]
        },
        "items": batch,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload_user, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    req = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": getattr(resp, "status", None), "body": json.loads(body)}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": str(exc), "body_text": body}
    except (URLError, TimeoutError) as exc:
        return {"ok": False, "status": None, "error": str(exc), "body_text": ""}


def apply_llm_item(row: dict[str, Any], item: dict[str, Any]) -> bool:
    sentiment = item.get("sentiment_label")
    stance = item.get("stance_label")
    if sentiment not in SENTIMENT_LABELS or stance not in STANCE_LABELS:
        return False
    s_conf = float(item.get("sentiment_confidence", item.get("confidence", 0.75)))
    a_conf = float(item.get("stance_confidence", item.get("confidence", 0.75)))
    row["llm_sentiment_label"] = sentiment
    row["llm_sentiment_confidence"] = round(max(0.4, min(s_conf, 0.95)), 2)
    row["llm_sentiment_reason"] = str(item.get("sentiment_reason", "")).strip()
    row["llm_stance_label"] = stance
    row["llm_stance_confidence"] = round(max(0.4, min(a_conf, 0.95)), 2)
    row["llm_stance_reason"] = str(item.get("stance_reason", "")).strip()
    row["sentiment_label"] = sentiment
    row["sentiment_confidence"] = row["llm_sentiment_confidence"]
    row["sentiment_reason"] = "LLM校正：" + row["llm_sentiment_reason"]
    row["stance_label"] = stance
    row["stance_confidence"] = row["llm_stance_confidence"]
    row["stance_reason"] = "LLM校正：" + row["llm_stance_reason"]
    row["label_method"] = "rule+llm"
    if (
        sentiment == "sarcastic"
        or stance == "unclear_meme"
        or row["sentiment_confidence"] < 0.70
        or row["stance_confidence"] < 0.70
        or int(row.get("like_count") or 0) >= 1000
    ):
        row["need_review"] = "true"
    else:
        row["need_review"] = "false"
    return True


def apply_llm(
    labeled_rows: list[dict[str, Any]],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    batch_size: int,
    timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    by_id = {row["content_id"]: row for row in labeled_rows}
    logs: list[dict[str, Any]] = []
    items = [
        {
            "content_id": row["content_id"],
            "source_title": row["source_title"],
            "topic_label": row["topic_label"],
            "content_text": compact_text(row["content_text"]),
            "rule_sentiment_label": row["rule_sentiment_label"],
            "rule_stance_label": row["rule_stance_label"],
        }
        for row in labeled_rows
    ]
    for start in range(0, len(items), batch_size):
        batch_no = start // batch_size + 1
        batch = items[start : start + batch_size]
        result = call_deepseek(api_key, endpoint, model, batch, timeout)
        raw_path = LLM_RAW_DIR / f"batch_{batch_no:03d}.json"
        write_json(raw_path, result)
        log = {
            "batch_no": batch_no,
            "item_count": len(batch),
            "ok": result.get("ok"),
            "status": result.get("status"),
            "applied": "",
            "raw_path": str(raw_path),
            "error": result.get("error", ""),
        }
        if result.get("ok"):
            try:
                content = result["body"]["choices"][0]["message"]["content"]
                parsed = extract_json_payload(content)
                llm_items = normalize_llm_items(parsed)
                applied = 0
                for item in llm_items:
                    cid = item.get("content_id")
                    if cid in by_id and apply_llm_item(by_id[cid], item):
                        applied += 1
                log["applied"] = applied
                log["item_count"] = len(llm_items)
            except Exception as exc:  # noqa: BLE001
                log["error"] = f"parse_or_apply_error: {exc}"
        logs.append(log)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    write_csv(LLM_RAW_DIR.parent / "llm_batches.csv", logs, ["batch_no", "item_count", "ok", "status", "applied", "raw_path", "error"])
    return labeled_rows


def apply_saved_llm(labeled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["content_id"]: row for row in labeled_rows}
    logs: list[dict[str, Any]] = []
    for raw_path in sorted(LLM_RAW_DIR.glob("batch_*.json")):
        match = re.search(r"batch_(\d+)", raw_path.name)
        batch_no = int(match.group(1)) if match else 0
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        log = {
            "batch_no": batch_no,
            "item_count": "",
            "ok": result.get("ok"),
            "status": result.get("status"),
            "applied": "",
            "raw_path": str(raw_path),
            "error": "",
        }
        if result.get("ok"):
            try:
                content = result["body"]["choices"][0]["message"]["content"]
                parsed = extract_json_payload(content)
                llm_items = normalize_llm_items(parsed)
                applied = 0
                for item in llm_items:
                    cid = item.get("content_id")
                    if cid in by_id and apply_llm_item(by_id[cid], item):
                        applied += 1
                log["item_count"] = len(llm_items)
                log["applied"] = applied
            except Exception as exc:  # noqa: BLE001
                log["error"] = f"parse_or_apply_error: {exc}"
        else:
            log["error"] = result.get("error", "")
        logs.append(log)
    write_csv(LLM_RAW_DIR.parent / "llm_batches.csv", logs, ["batch_no", "item_count", "ok", "status", "applied", "raw_path", "error"])
    return labeled_rows


def pct(n: int | float, total: int | float) -> float:
    return round(n / total * 100, 2) if total else 0.0


def summarize(rows_: list[dict[str, Any]], label_field: str) -> dict[str, Any]:
    counter = Counter(row[label_field] for row in rows_)
    likes = Counter()
    for row in rows_:
        likes[row[label_field]] += int(row.get("like_count") or 0)
    total = len(rows_)
    return {
        label: {
            "content_count": counter[label],
            "content_percent": pct(counter[label], total),
            "like_sum": likes[label],
        }
        for label in sorted(counter)
    }


def summary_by_source(rows_: list[dict[str, Any]], label_field: str) -> list[dict[str, Any]]:
    totals = Counter(row["source_id"] for row in rows_)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows_:
        key = (row["source_id"], row["source_title"], row[label_field])
        item = grouped.setdefault(
            key,
            {
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                label_field: row[label_field],
                "content_count": 0,
                "content_percent": 0,
                "like_sum": 0,
                "avg_like": 0,
                "max_like": 0,
            },
        )
        like = int(row.get("like_count") or 0)
        item["content_count"] += 1
        item["like_sum"] += like
        item["max_like"] = max(item["max_like"], like)
    result = []
    for item in grouped.values():
        item["content_percent"] = pct(item["content_count"], totals[item["source_id"]])
        item["avg_like"] = round(item["like_sum"] / item["content_count"], 2)
        result.append(item)
    result.sort(key=lambda x: (x["source_id"], -x["content_count"], str(x[label_field])))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Module 4 sentiment and stance analysis.")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--reuse-llm-raw", action="store_true")
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
    topic_rows = read_csv(MODULE3_PATH)
    labeled_rows: list[dict[str, Any]] = []
    for row in topic_rows:
        rule = rule_classify(row)
        labeled_rows.append(
            {
                "content_id": row["content_id"],
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                "content_type": row["content_type"],
                "content_text": row["content_text"],
                "topic_label": row["topic_label"],
                "created_at": row["created_at"],
                "like_count": row.get("like_count") or 0,
                "rule_sentiment_label": rule["sentiment_label"],
                "rule_sentiment_confidence": rule["sentiment_confidence"],
                "rule_sentiment_reason": rule["sentiment_reason"],
                "rule_stance_label": rule["stance_label"],
                "rule_stance_confidence": rule["stance_confidence"],
                "rule_stance_reason": rule["stance_reason"],
                "sentiment_label": rule["sentiment_label"],
                "sentiment_confidence": rule["sentiment_confidence"],
                "sentiment_reason": rule["sentiment_reason"],
                "stance_label": rule["stance_label"],
                "stance_confidence": rule["stance_confidence"],
                "stance_reason": rule["stance_reason"],
                "matched_keywords": rule["matched_keywords"],
                "sentiment_candidate_labels": rule["sentiment_candidate_labels"],
                "stance_candidate_labels": rule["stance_candidate_labels"],
                "llm_sentiment_label": "",
                "llm_sentiment_confidence": "",
                "llm_sentiment_reason": "",
                "llm_stance_label": "",
                "llm_stance_confidence": "",
                "llm_stance_reason": "",
                "need_review": rule["need_review"],
                "label_method": "rule",
            }
        )

    if args.reuse_llm_raw:
        labeled_rows = apply_saved_llm(labeled_rows)
    elif args.use_llm:
        api_key = deepseek_config.api_key
        if not api_key:
            raise SystemExit(
                f"Missing DEEPSEEK_API_KEY. Set it in environment or copy data/config/deepseek.example.env to {args.config}."
            )
        labeled_rows = apply_llm(
            labeled_rows,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            batch_size=args.batch_size,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        write_csv(MODULE_DIR / "llm_batches.csv", [], ["batch_no", "item_count", "ok", "status", "applied", "raw_path", "error"])

    content_fields = [
        "content_id",
        "source_id",
        "source_title",
        "content_type",
        "content_text",
        "topic_label",
        "created_at",
        "like_count",
        "rule_sentiment_label",
        "rule_sentiment_confidence",
        "rule_sentiment_reason",
        "rule_stance_label",
        "rule_stance_confidence",
        "rule_stance_reason",
        "sentiment_label",
        "sentiment_confidence",
        "sentiment_reason",
        "stance_label",
        "stance_confidence",
        "stance_reason",
        "matched_keywords",
        "sentiment_candidate_labels",
        "stance_candidate_labels",
        "llm_sentiment_label",
        "llm_sentiment_confidence",
        "llm_sentiment_reason",
        "llm_stance_label",
        "llm_stance_confidence",
        "llm_stance_reason",
        "need_review",
        "label_method",
    ]
    write_csv(MODULE_DIR / "content_sentiment_stance.csv", labeled_rows, content_fields)

    sentiment_source = summary_by_source(labeled_rows, "sentiment_label")
    stance_source = summary_by_source(labeled_rows, "stance_label")
    write_csv(
        MODULE_DIR / "sentiment_summary_by_source.csv",
        sentiment_source,
        ["source_id", "source_title", "sentiment_label", "content_count", "content_percent", "like_sum", "avg_like", "max_like"],
    )
    write_csv(
        MODULE_DIR / "stance_summary_by_source.csv",
        stance_source,
        ["source_id", "source_title", "stance_label", "content_count", "content_percent", "like_sum", "avg_like", "max_like"],
    )

    llm_batches_path = MODULE_DIR / "llm_batches.csv"
    llm_logs = read_csv(llm_batches_path) if llm_batches_path.exists() else []
    llm_applied = sum(int(log.get("applied") or 0) for log in llm_logs)
    llm_ok = sum(1 for log in llm_logs if str(log.get("ok")) == "True")

    overview = {
        "module": "module4_sentiment_stance",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_topics": str(MODULE3_PATH),
        "use_llm": args.use_llm or args.reuse_llm_raw,
        "reused_llm_raw": args.reuse_llm_raw,
        "model": args.model if (args.use_llm or args.reuse_llm_raw) else "",
        "content_count": len(labeled_rows),
        "sentiment_summary": summarize(labeled_rows, "sentiment_label"),
        "stance_summary": summarize(labeled_rows, "stance_label"),
        "rule_sentiment_summary": summarize(labeled_rows, "rule_sentiment_label"),
        "rule_stance_summary": summarize(labeled_rows, "rule_stance_label"),
        "need_review_count": sum(1 for row in labeled_rows if row["need_review"] == "true"),
        "llm_batches": llm_logs,
        "notes": [
            "当前版本暂跳过人工校验。",
            "最终标签优先采用 DeepSeek 校正结果；若未返回则保留规则初标。",
            "情绪和立场分开判断。",
        ],
    }
    write_json(MODULE_DIR / "sentiment_stance_overview.json", overview)

    sentiment_lines = "\n".join(
        f"{label}: {data['content_count']} 条，占比 {data['content_percent']}%，获赞 {data['like_sum']}"
        for label, data in sorted(overview["sentiment_summary"].items(), key=lambda item: item[1]["content_count"], reverse=True)
    )
    stance_lines = "\n".join(
        f"{label}: {data['content_count']} 条，占比 {data['content_percent']}%，获赞 {data['like_sum']}"
        for label, data in sorted(overview["stance_summary"].items(), key=lambda item: item[1]["content_count"], reverse=True)
    )
    top_liked = sorted(labeled_rows, key=lambda row: int(row.get("like_count") or 0), reverse=True)[:10]
    top_liked_lines = "\n".join(
        f"{row['like_count']}赞 | {row['sentiment_label']} | {row['stance_label']} | {shorten(row['content_text'], 90)}"
        for row in top_liked
    )
    summary_md = f"""# 模块 4：情绪与立场分析

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

## 情绪分布

```text
{sentiment_lines}
```

## 立场分布

```text
{stance_lines}
```

## 高赞评论标签抽样

```text
{top_liked_lines}
```

## 质量提示

```text
需要复核数量：{overview['need_review_count']}
```

当前结果保存在：

```text
data/analysis/module4_sentiment_stance/content_sentiment_stance.csv
data/analysis/module4_sentiment_stance/sentiment_summary_by_source.csv
data/analysis/module4_sentiment_stance/stance_summary_by_source.csv
data/analysis/module4_sentiment_stance/sentiment_stance_overview.json
```
"""
    (MODULE_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"content_sentiment_stance: {MODULE_DIR / 'content_sentiment_stance.csv'}")
    print(f"overview: {MODULE_DIR / 'sentiment_stance_overview.json'}")
    print(f"summary: {MODULE_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

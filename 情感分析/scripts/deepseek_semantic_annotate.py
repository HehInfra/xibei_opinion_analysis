from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "情感分析"
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "outputs"
RAW_DIR = OUTPUT_DIR / "deepseek_raw"
CONFIG_PATH = WORK_DIR / "config" / "deepseek.env"
LEGACY_CONFIG_PATH = ROOT / "data" / "config" / "deepseek.env"

DEFAULT_INPUT = DATA_DIR / "semantic_annotation_sample.csv"
ANNOTATED_JSONL = OUTPUT_DIR / "semantic_annotations_deepseek.jsonl"
ANNOTATED_CSV = OUTPUT_DIR / "semantic_annotations_deepseek.csv"
REVIEW_CSV = OUTPUT_DIR / "semantic_annotations_for_review.csv"


LABEL_GUIDE = """
你需要为 B 站西贝预制菜事件评论做五维语义标注。

主题 topic_label，只能选一个：
- prepared_food_authenticity：预制菜与食材真实性
- price_value：价格与性价比
- food_safety_child_meal：食品安全与儿童餐
- brand_pr_crisis：品牌公关与回应翻车
- consumer_right_to_know：消费者知情权
- luoyonghao_persona：罗永浩形象与维权叙事
- industry_trust：餐饮行业信任
- meme_offtopic：玩梗或跑题
- other_unclear：其他或不明确

立场对象 stance_target，只能选一个：
- xibei
- luoyonghao
- industry
- consumers
- unclear

立场方向 stance_label，只能选一个：
- support
- question
- neutral
- unclear

情绪 emotion_label，只能选一个：
- positive
- neutral
- negative
- angry
- disappointed
- anxious
- sarcastic

话语方式 discourse_labels，可多选，最多 3 个：
- rational_argument
- personal_experience
- information_claim
- questioning
- sarcasm
- meme
- echoing
- insult_attack
- simple_attitude

风险特征 risk_labels，可多选：
- emotional_amplification
- reputational_attack
- unverified_claim
- personal_attack
- group_mockery
- boycott_mobilization
- low_risk_discussion

强度 intensity：1-5 的整数。
置信度 confidence：0-1 的小数。
need_review：true 或 false。

注意：
1. 主题、立场、情绪分别只选一个主标签。
2. 话语方式和风险特征可以多选。
3. 如果没有明显风险，risk_labels 必须包含 low_risk_discussion。
4. 质疑西贝不等于风险，只有明显攻击、煽动、未经证实指控等才标风险。
5. 反讽和玩梗要按真实语义判断，不要只看字面。
6. annotation_reason 用一句中文简短说明即可。
"""


OUTPUT_SCHEMA = """
请只输出 JSON，不要输出 markdown。
输出必须是一个对象，格式如下：
{
  "items": [
    {
      "sample_id": "S0001",
      "content_id": "...",
      "topic_label": "...",
      "stance_target": "...",
      "stance_label": "...",
      "emotion_label": "...",
      "discourse_labels": ["..."],
      "risk_labels": ["..."],
      "intensity": 1,
      "confidence": 0.0,
      "need_review": false,
      "annotation_reason": "..."
    }
  ]
}
"""


CSV_FIELDS = [
    "sample_id",
    "sample_bucket",
    "content_id",
    "source_id",
    "source_title",
    "content_type",
    "user_id",
    "user_name",
    "content_text",
    "created_at",
    "like_count",
    "received_reply_count",
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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_csv_value(row.get(field, "")) for field in CSV_FIELDS})


def normalize_csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item).replace("\x00", "") for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value).replace("\x00", "")


def batched(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def build_messages(batch: list[dict[str, str]]) -> list[dict[str, str]]:
    comments = []
    for row in batch:
        comments.append(
            {
                "sample_id": row["sample_id"],
                "content_id": row["content_id"],
                "content_type": row["content_type"],
                "source_title": row["source_title"],
                "like_count": row["like_count"],
                "received_reply_count": row["received_reply_count"],
                "content_text": row["content_text"],
            }
        )
    user_content = "\n".join(
        [
            LABEL_GUIDE,
            OUTPUT_SCHEMA,
            "待标注评论如下：",
            json.dumps(comments, ensure_ascii=False, indent=2),
        ]
    )
    return [
        {
            "role": "system",
            "content": "你是中文互联网评论语义标注员，必须严格按照标签体系输出可解析 JSON。",
        },
        {"role": "user", "content": user_content},
    ]


def call_deepseek(api_key: str, model: str, messages: list[dict[str, str]], timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_item(item: dict[str, Any], row_by_sample: dict[str, dict[str, str]]) -> dict[str, Any]:
    sample_id = str(item.get("sample_id", ""))
    base = dict(row_by_sample.get(sample_id, {}))
    merged: dict[str, Any] = {**base, **item}

    if not isinstance(merged.get("discourse_labels"), list):
        merged["discourse_labels"] = split_labels(merged.get("discourse_labels", ""))
    if not isinstance(merged.get("risk_labels"), list):
        merged["risk_labels"] = split_labels(merged.get("risk_labels", ""))

    try:
        merged["intensity"] = int(merged.get("intensity", 0))
    except (TypeError, ValueError):
        merged["intensity"] = 0

    try:
        merged["confidence"] = float(merged.get("confidence", 0))
    except (TypeError, ValueError):
        merged["confidence"] = 0.0

    if isinstance(merged.get("need_review"), str):
        merged["need_review"] = merged["need_review"].lower() == "true"
    elif not isinstance(merged.get("need_review"), bool):
        merged["need_review"] = merged["confidence"] < 0.7

    return merged


def split_labels(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).replace(",", ";").split(";") if item.strip()]


def load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = []
    done = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                done.add(sample_id)
    return rows, done


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_raw(batch_index: int, request_messages: list[dict[str, str]], response_text: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_index": batch_index,
        "request_messages": request_messages,
        "response_text": response_text,
    }
    path = RAW_DIR / f"batch_{batch_index:04d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def next_raw_batch_index() -> int:
    if not RAW_DIR.exists():
        return 1
    indexes = []
    for path in RAW_DIR.glob("batch_*.json"):
        match = re.search(r"batch_(\d+)\.json$", path.name)
        if match:
            indexes.append(int(match.group(1)))
    return max(indexes, default=0) + 1


def call_with_retries(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            return call_deepseek(api_key, model, messages, timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
            if attempt > retries:
                break
            print(f"Request failed on attempt {attempt}, retrying in {retry_sleep}s: {exc}", flush=True)
            time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate semantic sample with DeepSeek.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Max rows to annotate. 0 means all.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file(CONFIG_PATH)
    load_env_file(LEGACY_CONFIG_PATH)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    rows = read_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]

    existing_rows, done_ids = load_existing(ANNOTATED_JSONL) if args.resume else ([], set())
    pending_rows = [row for row in rows if row["sample_id"] not in done_ids]

    if args.dry_run:
        messages = build_messages(pending_rows[: args.batch_size])
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        print(f"Pending rows: {len(pending_rows)}")
        return

    if not api_key:
        raise SystemExit(
            "Missing DEEPSEEK_API_KEY. Copy 情感分析/config/deepseek.example.env to "
            "情感分析/config/deepseek.env and fill the key, or export DEEPSEEK_API_KEY."
        )

    row_by_sample = {row["sample_id"]: row for row in rows}
    all_rows = list(existing_rows)

    batches = batched(pending_rows, args.batch_size)
    first_batch_index = next_raw_batch_index()
    for offset, batch in enumerate(batches, start=1):
        batch_index = first_batch_index + offset - 1
        messages = build_messages(batch)
        try:
            response_text = call_with_retries(
                api_key,
                model,
                messages,
                args.timeout,
                args.retries,
                args.retry_sleep,
            )
            write_raw(batch_index, messages, response_text)
            parsed = parse_json_response(response_text)
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Response JSON does not contain list field: items")
            annotated = [validate_item(item, row_by_sample) for item in items]
            append_jsonl(ANNOTATED_JSONL, annotated)
            all_rows.extend(annotated)
            done_ids.update(str(row.get("sample_id", "")) for row in annotated)
            write_csv(ANNOTATED_CSV, all_rows)
            write_csv(REVIEW_CSV, all_rows)
            print(f"Annotated batch {batch_index}: {len(annotated)} rows", flush=True)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
            print(f"Batch {batch_index} failed: {exc}", flush=True)
            raise
        time.sleep(args.sleep_seconds)

    write_csv(ANNOTATED_CSV, all_rows)
    review_rows = []
    for row in all_rows:
        copied = dict(row)
        if float(copied.get("confidence", 0) or 0) < 0.7:
            copied["need_review"] = True
        review_rows.append(copied)
    write_csv(REVIEW_CSV, review_rows)
    print(f"Wrote {len(all_rows)} annotations to {ANNOTATED_CSV}")
    print(f"Wrote review file to {REVIEW_CSV}")


if __name__ == "__main__":
    main()

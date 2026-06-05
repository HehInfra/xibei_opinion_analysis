from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
INTERIM_DIR = BASE_DIR / "interim"
CLEAN_JSON_DIR = INTERIM_DIR / "bilibili_clean_json"
LAYER1_DIR = BASE_DIR / "layer1"
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "xibei_event.db"


def ensure_dirs() -> None:
    CLEAN_JSON_DIR.mkdir(parents=True, exist_ok=True)
    LAYER1_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def parse_first_json(text: str) -> tuple[Any | None, str, str | None]:
    stripped = text.lstrip()
    if not stripped:
        return None, "", "empty_file"
    try:
        decoder = json.JSONDecoder()
        data, end_index = decoder.raw_decode(stripped)
        return data, stripped[end_index:].strip(), None
    except json.JSONDecodeError as exc:
        return None, "", str(exc)


def parse_video_meta(tail_text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in ("video_name", "video_time", "video_user_uid"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', tail_text)
        if match:
            meta[key] = match.group(1).strip()
    return meta


def parse_bilibili_time(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return str(value)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_video_time(value: str) -> str:
    if not value:
        return ""
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return value


def json_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def first_present(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def make_source_id(raw_file: Path, comment_json: dict[str, Any] | None, meta: dict[str, str]) -> str:
    oid = ""
    if comment_json:
        replies = ((comment_json.get("data") or {}).get("replies") or [])
        for reply in replies:
            if isinstance(reply, dict):
                oid = first_present(reply.get("oid_str"), reply.get("oid"))
                if oid:
                    break
    if oid:
        return f"bilibili_video:{oid}"
    title = meta.get("video_name") or raw_file.stem
    return f"bilibili_video:unknown_{stable_hash(title)}"


def get_member_user(member: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(member, dict):
        return None
    raw_user_id = first_present(member.get("mid"))
    if not raw_user_id:
        return None
    level_info = member.get("level_info") or {}
    return {
        "user_id": f"bilibili:{raw_user_id}",
        "platform": "bilibili",
        "raw_user_id": raw_user_id,
        "user_name": first_present(member.get("uname")),
        "user_type": "normal",
        "avatar_url": first_present(member.get("avatar")),
        "gender": first_present(member.get("sex")),
        "profile_text": first_present(member.get("sign")),
        "level": first_present(level_info.get("current_level")),
        "raw_data": json_compact(member),
    }


def update_user_seen(users: dict[str, dict[str, str]], user: dict[str, str], seen_time: str) -> None:
    user_id = user["user_id"]
    if user_id not in users:
        users[user_id] = {
            **user,
            "first_seen_time": seen_time,
            "last_seen_time": seen_time,
        }
        return
    current = users[user_id]
    if seen_time:
        if not current["first_seen_time"] or seen_time < current["first_seen_time"]:
            current["first_seen_time"] = seen_time
        if not current["last_seen_time"] or seen_time > current["last_seen_time"]:
            current["last_seen_time"] = seen_time
    for key in ("user_name", "avatar_url", "gender", "profile_text", "level", "raw_data"):
        if user.get(key):
            current[key] = user[key]


def relation_id(*parts: str) -> str:
    return "relation:" + stable_hash("|".join(parts), length=20)


def content_id_from_rpid(rpid: str) -> str:
    return f"bilibili_comment:{rpid}"


def flatten_replies(
    replies: list[Any],
    *,
    source_id: str,
    raw_file: Path,
    users: dict[str, dict[str, str]],
    contents: list[dict[str, str]],
    relations: list[dict[str, str]],
    parent_content: dict[str, str] | None = None,
) -> int:
    nested_count = 0
    for reply in replies:
        if not isinstance(reply, dict):
            continue

        member_user = get_member_user(reply.get("member"))
        if not member_user:
            continue

        created_at = parse_bilibili_time(reply.get("ctime"))
        update_user_seen(users, member_user, created_at)

        rpid = first_present(reply.get("rpid_str"), reply.get("rpid"))
        if not rpid:
            rpid = stable_hash(json_compact(reply), length=16)
        content_id = content_id_from_rpid(rpid)
        raw_parent = first_present(reply.get("parent_str"), reply.get("parent"))
        raw_root = first_present(reply.get("root_str"), reply.get("root"))
        content_text = first_present((reply.get("content") or {}).get("message"))
        content_type = "reply" if parent_content else "comment"

        parent_content_id = parent_content["content_id"] if parent_content else ""
        root_content_id = ""
        if parent_content:
            root_content_id = parent_content.get("root_content_id") or parent_content["content_id"]

        content_row = {
            "content_id": content_id,
            "platform": "bilibili",
            "source_id": source_id,
            "raw_content_id": rpid,
            "content_type": content_type,
            "user_id": member_user["user_id"],
            "content_text": content_text,
            "created_at": created_at,
            "like_count": first_present(reply.get("like")),
            "parent_content_id": parent_content_id,
            "root_content_id": root_content_id,
            "raw_file_path": str(raw_file),
            "raw_data": json_compact(reply),
        }
        contents.append(content_row)

        if parent_content:
            relations.append(
                {
                    "relation_id": relation_id(source_id, member_user["user_id"], "reply", parent_content["content_id"], content_id),
                    "platform": "bilibili",
                    "source_id": source_id,
                    "source_user_id": member_user["user_id"],
                    "target_user_id": parent_content["user_id"],
                    "relation_type": "reply",
                    "content_id": content_id,
                    "target_content_id": parent_content["content_id"],
                    "created_at": created_at,
                    "weight": "1.0",
                    "raw_file_path": str(raw_file),
                }
            )
        else:
            relations.append(
                {
                    "relation_id": relation_id(source_id, member_user["user_id"], "comment", content_id),
                    "platform": "bilibili",
                    "source_id": source_id,
                    "source_user_id": member_user["user_id"],
                    "target_user_id": "",
                    "relation_type": "comment_source",
                    "content_id": content_id,
                    "target_content_id": source_id,
                    "created_at": created_at,
                    "weight": "0.5",
                    "raw_file_path": str(raw_file),
                }
            )

        child_replies = reply.get("replies") or []
        if isinstance(child_replies, list) and child_replies:
            nested_count += len(child_replies)
            nested_count += flatten_replies(
                child_replies,
                source_id=source_id,
                raw_file=raw_file,
                users=users,
                contents=contents,
                relations=relations,
                parent_content=content_row,
            )
    return nested_count


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sqlite(tables: dict[str, tuple[list[dict[str, str]], list[str]]]) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, (rows, fieldnames) in tables.items():
            columns_sql = ", ".join(f"{field} TEXT" for field in fieldnames)
            conn.execute(f"CREATE TABLE {table_name} ({columns_sql})")
            if rows:
                placeholders = ", ".join("?" for _ in fieldnames)
                conn.executemany(
                    f"INSERT INTO {table_name} ({', '.join(fieldnames)}) VALUES ({placeholders})",
                    [[row.get(field, "") for field in fieldnames] for row in rows],
                )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ensure_dirs()

    audit_rows: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []
    users: dict[str, dict[str, str]] = {}
    contents: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    raw_records: list[dict[str, str]] = []
    video_meta_rows: list[dict[str, str]] = []

    for raw_file in sorted(RAW_DIR.glob("*.txt")):
        text = raw_file.read_text(encoding="utf-8", errors="replace")
        file_size = raw_file.stat().st_size
        comment_json, tail_text, parse_error = parse_first_json(text)
        meta = parse_video_meta(tail_text)
        source_id = make_source_id(raw_file, comment_json if isinstance(comment_json, dict) else None, meta)

        data = comment_json.get("data") if isinstance(comment_json, dict) else {}
        cursor = data.get("cursor") if isinstance(data, dict) else {}
        replies = data.get("replies") if isinstance(data, dict) else []
        if not isinstance(replies, list):
            replies = []
        pagination = cursor.get("pagination_reply") if isinstance(cursor, dict) else {}
        next_offset = pagination.get("next_offset") if isinstance(pagination, dict) else ""
        oid = ""
        for reply in replies:
            if isinstance(reply, dict):
                oid = first_present(reply.get("oid_str"), reply.get("oid"))
                if oid:
                    break

        nested_reply_count = 0
        error_message = parse_error or ""
        if isinstance(comment_json, dict):
            clean_json_path = CLEAN_JSON_DIR / f"{source_id.replace(':', '_')}.json"
            clean_json_path.write_text(json.dumps(comment_json, ensure_ascii=False, indent=2), encoding="utf-8")

            author_user_id = f"bilibili:{meta['video_user_uid']}" if meta.get("video_user_uid") else ""
            source_row = {
                "source_id": source_id,
                "platform": "bilibili",
                "source_type": "video",
                "source_title": meta.get("video_name") or raw_file.stem,
                "platform_source_id": oid,
                "source_url": "",
                "author_user_id": author_user_id,
                "published_at": normalize_video_time(meta.get("video_time", "")),
                "raw_file_path": str(raw_file),
                "comment_all_count": first_present(cursor.get("all_count") if isinstance(cursor, dict) else ""),
                "page_reply_count": str(len(replies)),
                "has_next_offset": "true" if next_offset else "false",
            }
            sources.append(source_row)
            video_meta_rows.append(
                {
                    **source_row,
                    "video_name": meta.get("video_name", ""),
                    "video_time": meta.get("video_time", ""),
                    "video_user_uid": meta.get("video_user_uid", ""),
                    "next_offset": first_present(next_offset),
                }
            )
            raw_records.append(
                {
                    "raw_record_id": f"raw:{stable_hash(str(raw_file))}",
                    "platform": "bilibili",
                    "source_id": source_id,
                    "raw_file_path": str(raw_file),
                    "file_name": raw_file.name,
                    "file_size": str(file_size),
                    "json_path": str(clean_json_path),
                    "tail_text": tail_text,
                    "parse_error": "",
                }
            )
            nested_reply_count = flatten_replies(
                replies,
                source_id=source_id,
                raw_file=raw_file,
                users=users,
                contents=contents,
                relations=relations,
            )
        else:
            raw_records.append(
                {
                    "raw_record_id": f"raw:{stable_hash(str(raw_file))}",
                    "platform": "bilibili",
                    "source_id": source_id,
                    "raw_file_path": str(raw_file),
                    "file_name": raw_file.name,
                    "file_size": str(file_size),
                    "json_path": "",
                    "tail_text": tail_text,
                    "parse_error": error_message,
                }
            )

        audit_rows.append(
            {
                "raw_file_path": str(raw_file),
                "file_name": raw_file.name,
                "file_size": str(file_size),
                "is_empty": "true" if file_size == 0 else "false",
                "can_extract_comment_json": "true" if isinstance(comment_json, dict) else "false",
                "has_video_meta": "true" if bool(meta) else "false",
                "video_name": meta.get("video_name", ""),
                "video_time": meta.get("video_time", ""),
                "video_user_uid": meta.get("video_user_uid", ""),
                "comment_all_count": first_present(cursor.get("all_count") if isinstance(cursor, dict) else ""),
                "page_reply_count": str(len(replies)),
                "has_next_offset": "true" if next_offset else "false",
                "nested_reply_count": str(nested_reply_count),
                "error_message": error_message,
            }
        )

    source_fieldnames = [
        "source_id",
        "platform",
        "source_type",
        "source_title",
        "platform_source_id",
        "source_url",
        "author_user_id",
        "published_at",
        "raw_file_path",
        "comment_all_count",
        "page_reply_count",
        "has_next_offset",
    ]
    user_fieldnames = [
        "user_id",
        "platform",
        "raw_user_id",
        "user_name",
        "user_type",
        "avatar_url",
        "gender",
        "profile_text",
        "level",
        "first_seen_time",
        "last_seen_time",
        "raw_data",
    ]
    content_fieldnames = [
        "content_id",
        "platform",
        "source_id",
        "raw_content_id",
        "content_type",
        "user_id",
        "content_text",
        "created_at",
        "like_count",
        "parent_content_id",
        "root_content_id",
        "raw_file_path",
        "raw_data",
    ]
    relation_fieldnames = [
        "relation_id",
        "platform",
        "source_id",
        "source_user_id",
        "target_user_id",
        "relation_type",
        "content_id",
        "target_content_id",
        "created_at",
        "weight",
        "raw_file_path",
    ]
    audit_fieldnames = [
        "raw_file_path",
        "file_name",
        "file_size",
        "is_empty",
        "can_extract_comment_json",
        "has_video_meta",
        "video_name",
        "video_time",
        "video_user_uid",
        "comment_all_count",
        "page_reply_count",
        "has_next_offset",
        "nested_reply_count",
        "error_message",
    ]
    raw_record_fieldnames = [
        "raw_record_id",
        "platform",
        "source_id",
        "raw_file_path",
        "file_name",
        "file_size",
        "json_path",
        "tail_text",
        "parse_error",
    ]
    video_meta_fieldnames = source_fieldnames + ["video_name", "video_time", "video_user_uid", "next_offset"]

    user_rows = sorted(users.values(), key=lambda row: row["user_id"])
    sources = sorted(sources, key=lambda row: row["source_id"])
    contents = sorted(contents, key=lambda row: row["content_id"])
    relations = sorted(relations, key=lambda row: row["relation_id"])

    write_csv(INTERIM_DIR / "raw_file_audit.csv", audit_rows, audit_fieldnames)
    write_csv(INTERIM_DIR / "bilibili_video_meta.csv", video_meta_rows, video_meta_fieldnames)
    write_csv(INTERIM_DIR / "raw_records.csv", raw_records, raw_record_fieldnames)
    write_csv(LAYER1_DIR / "sources.csv", sources, source_fieldnames)
    write_csv(LAYER1_DIR / "users.csv", user_rows, user_fieldnames)
    write_csv(LAYER1_DIR / "contents.csv", contents, content_fieldnames)
    write_csv(LAYER1_DIR / "relations.csv", relations, relation_fieldnames)

    write_sqlite(
        {
            "raw_records": (raw_records, raw_record_fieldnames),
            "sources": (sources, source_fieldnames),
            "users": (user_rows, user_fieldnames),
            "contents": (contents, content_fieldnames),
            "relations": (relations, relation_fieldnames),
            "raw_file_audit": (audit_rows, audit_fieldnames),
        }
    )

    print(f"raw files: {len(audit_rows)}")
    print(f"sources: {len(sources)}")
    print(f"users: {len(user_rows)}")
    print(f"contents: {len(contents)}")
    print(f"relations: {len(relations)}")
    print(f"audit: {INTERIM_DIR / 'raw_file_audit.csv'}")
    print(f"layer1: {LAYER1_DIR}")
    print(f"sqlite: {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


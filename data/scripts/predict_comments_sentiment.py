"""对指定的评论文件（JSONL）运行项目已微调的 MacBERT 情绪分类器。

复用 `train_macbert_sentiment.py` 训练出的模型（data/modeling/macbert_sentiment/），
对 `data/raw/Comments/Comments_5/` 下的评论逐条预测情绪标签，输出与全量预测
（sentiment_predictions.csv）完全一致的列结构，便于和已有结果拼接/对比。

情绪标签：positive / negative / neutral / sarcastic。

用法：
    python data/scripts/predict_comments_sentiment.py
    python data/scripts/predict_comments_sentiment.py --input <jsonl 或目录> --out <csv>
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parents[1]            # .../data
MODULE_DIR = BASE_DIR / "modeling"
FINETUNED_DIR = MODULE_DIR / "macbert_sentiment"
DEFAULT_INPUT = BASE_DIR / "raw" / "Comments" / "Comments_5"
DEFAULT_OUT = MODULE_DIR / "comments_5_sentiment_predictions.csv"

SENTIMENT_LABELS = ["positive", "negative", "neutral", "sarcastic"]
ID2LABEL = {i: label for i, label in enumerate(SENTIMENT_LABELS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_comments(input_path: Path) -> pd.DataFrame:
    """从 JSONL 文件（或目录下所有 .jsonl）读取评论，归一化为预测所需的列。"""
    if input_path.is_dir():
        files = sorted(glob.glob(str(input_path / "*.jsonl")))
    else:
        files = [str(input_path)]
    if not files:
        raise SystemExit(f"未找到任何 .jsonl 文件：{input_path}")

    rows: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                comment_id = str(obj.get("comment_id", "")).strip()
                video_id = str(obj.get("video_id", "")).strip()
                parent = str(obj.get("parent_comment_id", "0")).strip()
                rows.append({
                    "content_id": f"bilibili_comment:{comment_id}" if comment_id else "",
                    "source_id": f"bilibili_video:{video_id}" if video_id else "",
                    # 顶层评论 parent=0 记为 comment，其余为 reply（与 layer1 约定一致）
                    "content_type": "comment" if parent in ("", "0") else "reply",
                    "content_text": str(obj.get("content", "")).strip(),
                    "source_file": Path(fp).name,
                })
    df = pd.DataFrame(rows)
    df = df[df["content_text"] != ""].reset_index(drop=True)
    log(f"读取 {len(files)} 个文件，共 {len(df)} 条非空评论")
    return df


def predict(df: pd.DataFrame, batch_size: int, max_len: int, device: torch.device) -> pd.DataFrame:
    from transformers import BertForSequenceClassification, BertTokenizerFast

    if not (FINETUNED_DIR / "config.json").exists():
        raise SystemExit(f"未找到微调模型：{FINETUNED_DIR}，请先运行 train_macbert_sentiment.py。")
    tokenizer = BertTokenizerFast.from_pretrained(str(FINETUNED_DIR))
    model = BertForSequenceClassification.from_pretrained(str(FINETUNED_DIR)).to(device)
    model.eval()

    texts = df["content_text"].tolist()
    total = len(texts)
    softmax = nn.Softmax(dim=-1)
    all_probs: list[np.ndarray] = []
    log(f"开始预测 {total} 条（device={device}）")
    with torch.no_grad():
        for start in range(0, total, batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            all_probs.append(softmax(logits).cpu().numpy())
    probs = np.vstack(all_probs) if all_probs else np.empty((0, len(SENTIMENT_LABELS)))
    pred_ids = probs.argmax(axis=1)

    out = pd.DataFrame({
        "content_id": df["content_id"].values,
        "source_id": df["source_id"].values,
        "content_type": df["content_type"].values,
        "content_text": df["content_text"].values,
        "sentiment_label": [ID2LABEL[i] for i in pred_ids],
        "sentiment_confidence": probs.max(axis=1).round(4),
        "label_method": "macbert",
        "source_file": df["source_file"].values,
    })
    for i, label in enumerate(SENTIMENT_LABELS):
        out[f"p_{label}"] = probs[:, i].round(4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="对评论 JSONL 运行 MacBERT 情绪预测。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSONL 文件或包含 .jsonl 的目录。")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 CSV 路径。")
    parser.add_argument("--copy-to", default=None, help="额外再拷贝一份结果到该路径。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动检测。")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    df = load_comments(Path(args.input))
    out = predict(df, args.batch_size, args.max_len, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"预测完成，写入：{out_path}")
    log(f"标签分布：{out['sentiment_label'].value_counts().to_dict()}")

    if args.copy_to:
        import shutil

        copy_path = Path(args.copy_to)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, copy_path)
        log(f"已额外拷贝一份到：{copy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

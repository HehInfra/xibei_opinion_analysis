"""模块 4 第三步：MacBERT 微调情绪分类器。

按照《三合一文本标注与信息茧房分析方案》第 8 节的 "规则初标 + DeepSeek 校正 + MacBERT 微调"，
本脚本完成第三步：

1. 把本地 TF1 checkpoint (chinese_macbert_large) 一次性转换成 PyTorch/HF 模型（带缓存）。
2. 用 module4 已产出的 455 条 rule+DeepSeek 弱标注微调一个 4 类情绪分类器。
3. 在验证集上评估，并对全量评论 (data/layer1/contents.csv) 预测情绪标签。

情绪标签：positive / negative / neutral / sarcastic。

注意：训练数据仅 455 条且类别极不均衡，本版本定位为跑通闭环的原型，不是高精度标注模型。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[1]            # .../data
PROJECT_DIR = BASE_DIR.parent                             # project root
MODEL_SRC = PROJECT_DIR / "chinese_macbert_large"
HF_DIR = MODEL_SRC / "hf"                                 # 转换后的 PyTorch 模型缓存
TF_CKPT_PREFIX = MODEL_SRC / "chinese_macbert_large.ckpt"
TF_CONFIG_JSON = MODEL_SRC / "macbert_large_config.json"
VOCAB_TXT = MODEL_SRC / "vocab.txt"

TRAIN_CSV = BASE_DIR / "analysis" / "module4_sentiment_stance" / "content_sentiment_stance.csv"
CONTENTS_CSV = BASE_DIR / "layer1" / "contents.csv"
MODULE_DIR = BASE_DIR / "modeling"
FINETUNED_DIR = MODULE_DIR / "macbert_sentiment"

SENTIMENT_LABELS = ["positive", "negative", "neutral", "sarcastic"]
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

SEED = 42


_LOG_FH = None  # 由 main() 打开的 UTF-8 日志文件句柄


def log(msg: str) -> None:
    """带时间戳、立即 flush 的进度打印，便于实时监视日志文件。"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def gpu_mem(device) -> str:
    if device.type == "cuda":
        return f"gpu={torch.cuda.memory_allocated(device) / 1024**3:.1f}GB"
    return "cpu"


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- #
# 1. TF -> PyTorch 转换
# --------------------------------------------------------------------------- #
def ensure_pytorch_model() -> Path:
    """把本地 TF1 checkpoint 转换成 HF 格式（仅首次运行，带缓存）。"""
    from transformers import BertConfig

    has_weights = (HF_DIR / "model.safetensors").exists() or (HF_DIR / "pytorch_model.bin").exists()
    if has_weights and (HF_DIR / "config.json").exists() and (HF_DIR / "vocab.txt").exists():
        print(f"[convert] 已存在转换好的模型，跳过：{HF_DIR}")
        return HF_DIR

    if not TF_CKPT_PREFIX.with_suffix(".ckpt.index").exists() and not (MODEL_SRC / "chinese_macbert_large.ckpt.index").exists():
        raise SystemExit(f"未找到 TF checkpoint：{TF_CKPT_PREFIX}.*")

    print("[convert] 读取 TF1 checkpoint 并转换为 PyTorch（需要 tensorflow，仅首次）...")
    try:
        from transformers import BertForPreTraining
        from transformers.models.bert.modeling_bert import load_tf_weights_in_bert
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"导入 transformers BERT 失败：{exc}")

    config = BertConfig.from_json_file(str(TF_CONFIG_JSON))
    model = BertForPreTraining(config)
    try:
        load_tf_weights_in_bert(model, config, str(TF_CKPT_PREFIX))
    except ImportError as exc:
        raise SystemExit(
            "转换 TF checkpoint 需要安装 tensorflow：pip install tensorflow-cpu\n"
            f"原始错误：{exc}"
        )

    HF_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(HF_DIR)
    shutil.copyfile(VOCAB_TXT, HF_DIR / "vocab.txt")
    # 写一个最小 tokenizer 配置，保证 BertTokenizerFast 可直接加载
    (HF_DIR / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True, "model_max_length": 512, "tokenizer_class": "BertTokenizer"}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[convert] 转换完成，已保存到：{HF_DIR}")
    return HF_DIR


# --------------------------------------------------------------------------- #
# 2. 数据集
# --------------------------------------------------------------------------- #
class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_training_frame(train_csv: Path = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    df = df[["content_text", "sentiment_label"]].copy()
    df["content_text"] = df["content_text"].fillna("").str.strip()
    df = df[(df["content_text"] != "") & (df["sentiment_label"].isin(SENTIMENT_LABELS))]
    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. 训练与评估
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            golds.extend(labels.tolist())
    return golds, preds


def train(args) -> dict:
    set_seed(SEED)
    from transformers import BertForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup

    train_csv = Path(args.train_csv) if args.train_csv else TRAIN_CSV

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[train] device = {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    hf_dir = ensure_pytorch_model()
    tokenizer = BertTokenizerFast.from_pretrained(str(hf_dir))

    df = load_training_frame(train_csv)
    print(f"[train] 训练样本：{len(df)} 条；类别分布：{df['sentiment_label'].value_counts().to_dict()}")
    labels = df["sentiment_label"].map(LABEL2ID).tolist()
    texts = df["content_text"].tolist()

    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=args.val_size, random_state=SEED, stratify=labels
    )

    tr_enc = tokenizer(tr_texts, truncation=True, padding="max_length", max_length=args.max_len)
    va_enc = tokenizer(va_texts, truncation=True, padding="max_length", max_length=args.max_len)
    tr_ds = TextDataset(tr_enc, tr_labels)
    va_ds = TextDataset(va_enc, va_labels)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    # 逆频率类别权重，缓解极端不均衡
    counts = np.bincount(tr_labels, minlength=len(SENTIMENT_LABELS)).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(SENTIMENT_LABELS) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    print(f"[train] 类别权重：{dict(zip(SENTIMENT_LABELS, weights.round(3)))}")

    model = BertForSequenceClassification.from_pretrained(
        str(hf_dir), num_labels=len(SENTIMENT_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1 = -1.0
    best_report: dict = {}
    best_state: dict | None = None
    steps_per_epoch = len(tr_loader)
    report_every = max(1, steps_per_epoch // 10)
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.time()
        for step, batch in enumerate(tr_loader, 1):
            labels_b = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % report_every == 0 or step == steps_per_epoch:
                elapsed = time.time() - epoch_start
                eta = elapsed / step * (steps_per_epoch - step)
                pct = step / steps_per_epoch * 100
                log(
                    f"[train] epoch {epoch}/{args.epochs} step {step}/{steps_per_epoch} ({pct:.0f}%) "
                    f"loss={running/step:.4f} eta={fmt_eta(eta)} {gpu_mem(device)}"
                )
        golds, preds = evaluate(model, va_loader, device)
        macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
        acc = float(np.mean(np.array(golds) == np.array(preds)))
        log(f"[train] >>> epoch {epoch}/{args.epochs} done  loss={running/steps_per_epoch:.4f}  val_acc={acc:.3f}  val_macroF1={macro_f1:.3f}  (epoch {fmt_eta(time.time()-epoch_start)})")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            # 把最佳权重留在内存（CPU），训练结束后只保存一次，避免反复覆盖
            # 已被内存映射的 safetensors 文件（Windows os error 1224）。
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            report = classification_report(
                golds, preds, labels=list(range(len(SENTIMENT_LABELS))),
                target_names=SENTIMENT_LABELS, output_dict=True, zero_division=0,
            )
            cm = confusion_matrix(golds, preds, labels=list(range(len(SENTIMENT_LABELS)))).tolist()
            best_report = {
                "best_epoch": epoch,
                "val_accuracy": round(acc, 4),
                "val_macro_f1": round(macro_f1, 4),
                "classification_report": report,
                "confusion_matrix": cm,
                "confusion_matrix_labels": SENTIMENT_LABELS,
                "train_size": len(tr_labels),
                "val_size": len(va_labels),
            }

    # 训练结束后，载入最佳权重并只保存一次。用 safe_serialization=False（torch.save，
    # 不走 mmap）避免 Windows 上覆盖已映射 safetensors 文件时的 os error 1224。
    if best_state is not None:
        model.load_state_dict(best_state)
    if FINETUNED_DIR.exists():
        shutil.rmtree(FINETUNED_DIR, ignore_errors=True)
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINETUNED_DIR, safe_serialization=False)
    tokenizer.save_pretrained(FINETUNED_DIR)
    log(f"[train] 最佳验证 macro-F1 = {best_f1:.3f}，总耗时 {fmt_eta(time.time()-train_start)}，模型已保存到 {FINETUNED_DIR}")
    return best_report


# --------------------------------------------------------------------------- #
# 4. 全量预测
# --------------------------------------------------------------------------- #
def predict_all(args) -> int:
    from transformers import BertForSequenceClassification, BertTokenizerFast

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (FINETUNED_DIR / "config.json").exists():
        raise SystemExit(f"未找到微调模型：{FINETUNED_DIR}，请先训练。")
    tokenizer = BertTokenizerFast.from_pretrained(str(FINETUNED_DIR))
    model = BertForSequenceClassification.from_pretrained(str(FINETUNED_DIR)).to(device)
    model.eval()

    df = pd.read_csv(CONTENTS_CSV, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit).copy()
    texts = df["content_text"].fillna("").str.strip().tolist()
    total = len(texts)
    log(f"[predict] 待预测评论：{total} 条（device={device}）")

    softmax = nn.Softmax(dim=-1)
    all_probs: list[np.ndarray] = []
    bs = args.batch_size
    n_batches = (total + bs - 1) // bs
    report_every = max(1, n_batches // 20)
    pred_start = time.time()
    with torch.no_grad():
        for bi, start in enumerate(range(0, total, bs), 1):
            chunk = texts[start : start + bs]
            enc = tokenizer(chunk, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            all_probs.append(softmax(logits).cpu().numpy())
            if bi % report_every == 0 or bi == n_batches:
                done = min(start + bs, total)
                elapsed = time.time() - pred_start
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                log(f"[predict] {done}/{total} ({done/total*100:.0f}%) {rate:.0f} rows/s eta={fmt_eta(eta)} {gpu_mem(device)}")
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
    })
    for i, label in enumerate(SENTIMENT_LABELS):
        out[f"p_{label}"] = probs[:, i].round(4)

    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODULE_DIR / "sentiment_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[predict] 预测完成，耗时 {fmt_eta(time.time()-pred_start)}，写入：{out_path}")
    dist = out["sentiment_label"].value_counts().to_dict()
    log(f"[predict] 预测标签分布：{dist}")
    return dist


# --------------------------------------------------------------------------- #
# 5. 汇总输出
# --------------------------------------------------------------------------- #
def write_outputs(eval_report: dict, pred_dist: dict | None) -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    (MODULE_DIR / "sentiment_eval.json").write_text(
        json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cr = eval_report.get("classification_report", {})
    per_class = "\n".join(
        f"{label}: precision {cr.get(label, {}).get('precision', 0):.2f}  "
        f"recall {cr.get(label, {}).get('recall', 0):.2f}  "
        f"f1 {cr.get(label, {}).get('f1-score', 0):.2f}  "
        f"support {int(cr.get(label, {}).get('support', 0))}"
        for label in SENTIMENT_LABELS
    )
    dist_lines = (
        "\n".join(f"{k}: {v} 条" for k, v in sorted((pred_dist or {}).items(), key=lambda x: -x[1]))
        if pred_dist else "（本次未执行全量预测）"
    )
    summary = f"""# 模块 4 扩展：MacBERT 情绪分类

## 方法

```text
规则初标 + DeepSeek 校正  ->  生成 455 条弱标注
弱标注  ->  微调 chinese_macbert_large（本地 TF checkpoint 转换为 PyTorch）
微调模型  ->  对全量评论预测情绪标签
```

情绪标签：positive / negative / neutral / sarcastic。

## 验证集评估

```text
best_epoch: {eval_report.get('best_epoch', '-')}
train_size: {eval_report.get('train_size', '-')}    val_size: {eval_report.get('val_size', '-')}
accuracy:  {eval_report.get('val_accuracy', '-')}
macro_f1:  {eval_report.get('val_macro_f1', '-')}
```

每类指标：

```text
{per_class}
```

## 全量预测标签分布

```text
{dist_lines}
```

## 输出文件

```text
data/modeling/macbert_sentiment/         微调后的模型
data/modeling/sentiment_predictions.csv  全量评论情绪预测
data/modeling/sentiment_eval.json        验证集评估指标
```

## 已知限制

```text
1. 训练数据仅 455 条弱标注，类别极不均衡（positive 仅 8 条），large 模型易过拟合。
2. 标签来自规则+DeepSeek，未经人工校验。
3. 本版本定位为跑通闭环的原型，不是高精度标注模型。
   后续应按方案第 11 节扩充到 2000+ 条标注再重训。
```
"""
    (MODULE_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[summary] 写入：{MODULE_DIR / 'summary.md'} 与 sentiment_eval.json")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="MacBERT 情绪分类：转换 + 微调 + 全量预测。")
    parser.add_argument("--convert-only", action="store_true", help="只做 TF->PyTorch 转换后退出。")
    parser.add_argument("--no-train", action="store_true", help="跳过微调（复用已训练模型）。")
    parser.add_argument("--no-predict-all", action="store_true", help="跳过全量预测。")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动检测。")
    parser.add_argument("--limit", type=int, default=0, help="只预测前 N 条（调试用）。")
    parser.add_argument("--train-csv", default=None,
                        help="训练数据 CSV（含 content_text/sentiment_label），默认用 module4 的 455 条。")
    args = parser.parse_args()

    global _LOG_FH
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(MODULE_DIR / "train_run.log", "w", encoding="utf-8")  # noqa: SIM115

    if args.convert_only:
        ensure_pytorch_model()
        return 0

    eval_report: dict = {}
    if not args.no_train:
        eval_report = train(args)
    elif (MODULE_DIR / "sentiment_eval.json").exists():
        eval_report = json.loads((MODULE_DIR / "sentiment_eval.json").read_text(encoding="utf-8"))

    pred_dist = None
    if not args.no_predict_all:
        pred_dist = predict_all(args)

    write_outputs(eval_report, pred_dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

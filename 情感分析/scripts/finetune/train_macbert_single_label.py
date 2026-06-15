from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_DIR = Path(__file__).resolve().parents[3]
SEMANTIC_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "hfl/chinese-macbert-base"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TASK_CONFIGS = {
    "topic": {
        "data": SEMANTIC_DIR / "data" / "training" / "topic.csv",
        "label_column": "topic_label",
        "output_dir": SEMANTIC_DIR / "models" / "finetuned" / "topic",
    },
    "emotion": {
        "data": SEMANTIC_DIR / "data" / "training" / "emotion.csv",
        "label_column": "emotion_label",
        "output_dir": SEMANTIC_DIR / "models" / "finetuned" / "emotion",
    },
    "stance_target": {
        "data": SEMANTIC_DIR / "data" / "training" / "stance_target.csv",
        "label_column": "stance_target",
        "output_dir": SEMANTIC_DIR / "models" / "finetuned" / "stance_target",
    },
    "stance": {
        "data": SEMANTIC_DIR / "data" / "training" / "stance.csv",
        "label_column": "stance_label",
        "output_dir": SEMANTIC_DIR / "models" / "finetuned" / "stance",
    },
}


@dataclass(frozen=True)
class SingleLabelExample:
    content_id: str
    text: str
    label: int
    split: str


class SingleLabelDataset:
    def __init__(self, examples: list[SingleLabelExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SingleLabelExample:
        return self.examples[index]


def read_rows(path: Path, label_column: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"content_id", "content_text", label_column, "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"训练数据缺少字段: {sorted(missing)}")
        return list(reader)


def build_label_mapping(rows: list[dict[str, str]], label_column: str) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({(row.get(label_column) or "").strip() for row in rows if (row.get(label_column) or "").strip()})
    if not labels:
        raise ValueError(f"训练数据中没有可用 {label_column}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def make_examples(
    rows: list[dict[str, str]],
    *,
    label_column: str,
    label_to_id: dict[str, int],
) -> list[SingleLabelExample]:
    examples: list[SingleLabelExample] = []
    for row in rows:
        text = (row.get("content_text") or "").strip()
        label_text = (row.get(label_column) or "").strip()
        split = (row.get("split") or "").strip()
        if not text or label_text not in label_to_id:
            continue
        examples.append(
            SingleLabelExample(
                content_id=row.get("content_id") or "",
                text=text,
                label=label_to_id[label_text],
                split=split,
            )
        )
    if not examples:
        raise ValueError("没有读到可训练样本")
    return examples


def limit_examples(examples: list[SingleLabelExample], limit: int | None) -> list[SingleLabelExample]:
    if limit is None:
        return examples
    if limit <= 0:
        raise ValueError("--limit-per-split 必须大于 0")
    return examples[:limit]


def split_examples(
    examples: list[SingleLabelExample],
    *,
    limit_per_split: int | None,
) -> tuple[list[SingleLabelExample], list[SingleLabelExample], list[SingleLabelExample]]:
    train = limit_examples([example for example in examples if example.split == "train"], limit_per_split)
    valid = limit_examples([example for example in examples if example.split == "valid"], limit_per_split)
    test = limit_examples([example for example in examples if example.split == "test"], limit_per_split)
    if not train:
        raise ValueError("训练集为空，请检查 split=train 的样本")
    if not valid:
        raise ValueError("验证集为空，请检查 split=valid 的样本")
    return train, valid, test


def label_counts(examples: list[SingleLabelExample], id_to_label: dict[int, str]) -> dict[str, int]:
    counts = {label: 0 for label in id_to_label.values()}
    for example in examples:
        counts[id_to_label[example.label]] += 1
    return counts


def choose_device(device_name: str) -> Any:
    import torch

    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境不可用 CUDA，请在 4090 机器上运行或改用 --device auto")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("当前环境不可用 MPS，请改用 --device auto 或 --device cpu")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("--device 只支持 auto/cuda/mps/cpu")


def device_status() -> dict[str, bool]:
    import torch

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(tokenizer: Any, max_length: int):
    import torch

    def collate(examples: list[SingleLabelExample]) -> dict[str, Any]:
        encoded = tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([example.label for example in examples], dtype=torch.long)
        return encoded

    return collate


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def run_train_epoch(*, model: Any, loader: Any, optimizer: Any, scheduler: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def evaluate(*, model: Any, loader: Any, device: Any, num_labels: int) -> dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    per_label = {label_id: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label_id in range(num_labels)}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]

            batch_size = int(labels.shape[0])
            total_loss += float(outputs.loss.detach().cpu()) * batch_size
            total_items += batch_size
            correct += int((predictions == labels).sum().detach().cpu())

            for gold, pred in zip(labels.detach().cpu().tolist(), predictions.detach().cpu().tolist(), strict=True):
                per_label[int(gold)]["support"] += 1
                if gold == pred:
                    per_label[int(gold)]["tp"] += 1
                else:
                    per_label[int(pred)]["fp"] += 1
                    per_label[int(gold)]["fn"] += 1

    label_metrics: dict[int, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label_id, stats in per_label.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        label_metrics[label_id] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": stats["support"],
        }

    return {
        "loss": round(total_loss / max(total_items, 1), 6),
        "accuracy": round(correct / max(total_items, 1), 6),
        "macro_f1": round(sum(f1_values) / max(len(f1_values), 1), 6),
        "examples": total_items,
        "per_label": label_metrics,
    }


def add_label_names(metrics: dict[str, Any], id_to_label: dict[int, str]) -> dict[str, Any]:
    copied = dict(metrics)
    copied["per_label"] = {id_to_label[int(label_id)]: values for label_id, values in metrics["per_label"].items()}
    return copied


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_defaults(task: str) -> dict[str, Any]:
    if task not in TASK_CONFIGS:
        raise ValueError(f"未知任务: {task}，可选任务: {sorted(TASK_CONFIGS)}")
    return TASK_CONFIGS[task]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MacBERT for one single-label semantic task.")
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), help="单标签任务名；任务入口脚本会自动传入")
    parser.add_argument("--data", default=None, help="训练 CSV 路径；默认按 --task 选择")
    parser.add_argument("--label-column", default=None, help="标签列名；默认按 --task 选择")
    parser.add_argument("--output-dir", default=None, help="模型保存目录；默认按 --task 选择")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="训练设备")
    parser.add_argument("--epochs", type=int, default=3, help="正式训练默认 3 epoch")
    parser.add_argument("--batch-size", type=int, default=16, help="4090 可先用 16；显存宽裕可调 32")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2, help="Linux/4090 可用 2-4；macOS/MPS 建议 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=None, help="每个 split 只取前 N 条，用于烟测")
    parser.add_argument("--from-tf", action="store_true", help="尝试从 TensorFlow checkpoint 加载")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"缺少 Python 依赖: {missing}\n"
            "请先安装训练依赖：\n"
            "  conda env create -f 情感分析/scripts/finetune/environment.yml\n"
            "或在已有环境中安装：\n"
            "  pip install -r 情感分析/scripts/finetune/requirements.txt"
        ) from exc

    if args.task is None:
        raise SystemExit("缺少 --task。可直接使用 train_macbert_topic.py / train_macbert_emotion.py 等任务入口脚本。")

    defaults = task_defaults(args.task)
    data_path = Path(args.data) if args.data else defaults["data"]
    label_column = args.label_column or defaults["label_column"]
    output_dir = Path(args.output_dir) if args.output_dir else defaults["output_dir"]

    seed_everything(args.seed)
    rows = read_rows(data_path, label_column)
    label_to_id, id_to_label = build_label_mapping(rows, label_column)
    examples = make_examples(rows, label_column=label_column, label_to_id=label_to_id)
    train_examples, valid_examples, test_examples = split_examples(examples, limit_per_split=args.limit_per_split)

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        from_tf=args.from_tf,
    )
    model.to(device)

    collate = collate_batch(tokenizer, args.max_length)
    train_loader = DataLoader(
        SingleLabelDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        SingleLabelDataset(valid_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = (
        DataLoader(
            SingleLabelDataset(test_examples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        if test_examples
        else None
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"workspace: {WORKSPACE_DIR}")
    print(f"task: {args.task}")
    print(f"data: {data_path}")
    print(f"label column: {label_column}")
    print(f"model: {args.model_name}")
    print(f"output: {output_dir}")
    print(f"device: {device}")
    print(f"device status: {device_status()}")
    print(f"torch: {torch.__version__}")
    print(f"train examples: {len(train_examples)} {label_counts(train_examples, id_to_label)}")
    print(f"valid examples: {len(valid_examples)} {label_counts(valid_examples, id_to_label)}")
    print(f"test examples: {len(test_examples)} {label_counts(test_examples, id_to_label)}")

    history: list[dict[str, Any]] = []
    best_valid_macro_f1 = -1.0
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        valid_metrics = add_label_names(
            evaluate(model=model, loader=valid_loader, device=device, num_labels=len(label_to_id)),
            id_to_label,
        )
        elapsed = round(time.time() - epoch_start, 2)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "valid": valid_metrics,
                "seconds": elapsed,
            }
        )
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f} "
            f"seconds={elapsed}"
        )

        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    final_test_metrics = None
    if test_loader is not None:
        final_test_metrics = add_label_names(
            evaluate(model=model, loader=test_loader, device=device, num_labels=len(label_to_id)),
            id_to_label,
        )
        print(
            f"test_loss={final_test_metrics['loss']:.4f} "
            f"test_acc={final_test_metrics['accuracy']:.4f} "
            f"test_macro_f1={final_test_metrics['macro_f1']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "label_mapping.json",
        {
            "label_to_id": label_to_id,
            "id_to_label": {str(key): value for key, value in id_to_label.items()},
        },
    )
    write_json(
        output_dir / "training_summary.json",
        {
            "task": args.task,
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": args.model_name,
            "from_tf": args.from_tf,
            "data_path": str(data_path),
            "label_column": label_column,
            "output_dir": str(output_dir),
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "limit_per_split": args.limit_per_split,
            "train_examples": len(train_examples),
            "valid_examples": len(valid_examples),
            "test_examples": len(test_examples),
            "train_label_counts": label_counts(train_examples, id_to_label),
            "valid_label_counts": label_counts(valid_examples, id_to_label),
            "test_label_counts": label_counts(test_examples, id_to_label),
            "best_valid_macro_f1": round(best_valid_macro_f1, 6),
            "history": history,
            "test": final_test_metrics,
        },
    )
    print(f"best model and summaries saved to: {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

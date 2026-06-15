from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEMANTIC_DIR = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = SEMANTIC_DIR / "data" / "semantic_comments_full.csv"
DEFAULT_OUTPUT = SEMANTIC_DIR / "outputs" / "full_semantic_predictions.csv"

DEFAULT_MODEL_DIRS = {
    "topic": SEMANTIC_DIR / "models" / "finetuned" / "topic",
    "emotion": SEMANTIC_DIR / "models" / "finetuned" / "emotion",
    "stance_target": SEMANTIC_DIR / "models" / "finetuned" / "stance_target",
    "stance": SEMANTIC_DIR / "models" / "finetuned" / "stance",
}


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


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        if "content_text" not in fieldnames:
            raise ValueError(f"输入文件缺少 content_text 字段: {path}")
        return list(reader), fieldnames


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(DEFAULT_MODEL_DIRS))
    if unknown:
        raise ValueError(f"未知任务: {unknown}，可选任务: {sorted(DEFAULT_MODEL_DIRS)}")
    if not tasks:
        raise ValueError("--tasks 不能为空")
    return tasks


def task_model_dir(args: argparse.Namespace, task: str) -> Path:
    override = getattr(args, f"{task}_model_dir")
    return Path(override) if override else DEFAULT_MODEL_DIRS[task]


def load_label_mapping(model_dir: Path) -> dict[int, str] | None:
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.exists():
        return None
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    id_to_label = payload.get("id_to_label") or {}
    return {int(key): value for key, value in id_to_label.items()}


def load_model(model_dir: Path, device: Any) -> tuple[Any, Any, dict[int, str]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    id_to_label = load_label_mapping(model_dir)
    if id_to_label is None:
        id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    return tokenizer, model, id_to_label


def predict_task(
    *,
    task: str,
    rows: list[dict[str, str]],
    model_dir: Path,
    device: Any,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model, id_to_label = load_model(model_dir, device)
    predictions: list[dict[str, Any]] = []
    texts = [(row.get("content_text") or "").strip() for row in rows]

    print(f"predict task={task} model_dir={model_dir} rows={len(rows)}")
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            scores, label_ids = torch.max(probabilities, dim=-1)

        for label_id, score in zip(label_ids.detach().cpu().tolist(), scores.detach().cpu().tolist(), strict=True):
            predictions.append(
                {
                    f"pred_{task}_label": id_to_label[int(label_id)],
                    f"pred_{task}_confidence": round(float(score), 6),
                }
            )

        finished = min(start + batch_size, len(texts))
        if finished == len(texts) or finished % max(batch_size * 20, 1) == 0:
            print(f"  {task}: {finished}/{len(texts)}")

    return predictions


def merge_predictions(rows: list[dict[str, Any]], task_predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output_rows = [dict(row) for row in rows]
    for task, predictions in task_predictions.items():
        if len(predictions) != len(output_rows):
            raise ValueError(f"{task} 预测条数不一致: {len(predictions)} != {len(output_rows)}")
        for row, prediction in zip(output_rows, predictions, strict=True):
            row.update(prediction)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict semantic labels for full comment table with fine-tuned models.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="全量评论 CSV，默认 semantic_comments_full.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="预测输出 CSV")
    parser.add_argument("--tasks", default="topic,emotion,stance_target,stance", help="逗号分隔任务列表")
    parser.add_argument("--topic-model-dir", default=None)
    parser.add_argument("--emotion-model-dir", default=None)
    parser.add_argument("--stance-target-model-dir", default=None)
    parser.add_argument("--stance-model-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="只预测前 N 条，用于烟测")
    return parser.parse_args()


def main() -> int:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"缺少 Python 依赖: {missing}\n"
            "请先安装：pip install -r 情感分析/scripts/finetune/requirements.txt"
        ) from exc

    args = parse_args()
    tasks = parse_tasks(args.tasks)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows, input_fieldnames = read_rows(input_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        rows = rows[: args.limit]

    device = choose_device(args.device)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"device: {device}")
    print(f"rows: {len(rows)}")
    print(f"tasks: {tasks}")

    task_predictions: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        task_predictions[task] = predict_task(
            task=task,
            rows=rows,
            model_dir=task_model_dir(args, task),
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    output_rows = merge_predictions(rows, task_predictions)
    prediction_fields: list[str] = []
    for task in tasks:
        prediction_fields.extend([f"pred_{task}_label", f"pred_{task}_confidence"])
    write_rows(output_path, output_rows, [*input_fieldnames, *prediction_fields])
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

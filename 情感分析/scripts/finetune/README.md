# 微调脚本

当前支持四个单标签分类微调任务：

- `topic`
- `emotion`
- `stance_target`
- `stance`

## 依赖

```bash
conda env create -f 情感分析/scripts/finetune/environment.yml
conda activate xibei-semantic-ft
```

## MacBook Air M4 烟测

首次运行如果使用 `hfl/chinese-macbert-base`，需要能访问 Hugging Face 下载模型。

```bash
conda run -n xibei-semantic-ft python 情感分析/scripts/finetune/train_macbert_topic.py \
  --device auto \
  --epochs 1 \
  --batch-size 4 \
  --limit-per-split 32
```

## 4090 正式训练

```bash
conda run -n xibei-semantic-ft python 情感分析/scripts/finetune/train_macbert_topic.py \
  --device cuda \
  --epochs 3 \
  --batch-size 16 \
  --max-length 128
```

更多服务器训练命令见：

```text
情感分析/docs/4090单标签微调说明.md
```

输出默认保存到：

```text
情感分析/models/finetuned/<task>/
```

训练完成后，用 `predict_semantic_full.py` 对全量评论预测：

```bash
conda run -n xibei-semantic-ft python 情感分析/scripts/finetune/predict_semantic_full.py \
  --device cuda \
  --batch-size 64 \
  --output 情感分析/outputs/predictions/full_semantic_predictions.csv
```

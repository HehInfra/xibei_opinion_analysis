# 情感分析与五维语义微调工作区

本文件夹是西贝预制菜事件 B 站评论的五维语义分析工作区。后续如果新开对话，先阅读本 README，即可知道当前进度和下一步要做什么。

## 0. 当前状态

已经完成：

```text
1. 从原项目数据库导出 40803 条评论全量底表
2. 从全量评论中抽样 1500 条
3. 使用 DeepSeek API 对 1500 条样本做五维语义标注
4. 生成可用于微调的伪标签训练集
5. 将本地 MacBERT 基座模型移动到本文件夹中
6. 完成 topic、emotion、stance_target、stance 四个单标签分类器微调
7. 对 40803 条全量评论完成四维语义预测
8. 完成四维语义与源数据交叉分析，报告见 `docs/四维语义交叉分析执行报告.md`
```

当前决定：

```text
当前四个单标签分类器已经跑通本地微调和全量预测；第五、六维的 discourse/risk 暂未正式训练。
```

因此，后续模型评估要注意：

```text
验证集和测试集同样来自 DeepSeek 伪标签。
模型评估的是“拟合 DeepSeek 标注规则的能力”，不是严格人工金标准准确率。
```

## 1. 项目目标

本工作区要完成的任务不是单纯情绪三分类，而是五维评论语义分类：

```text
主题：评论在讨论什么
立场对象：评论主要针对谁
立场方向：评论支持、质疑还是中立
情绪：评论表达什么情绪
话语方式：评论如何表达，例如玩梗、反讽、质问
风险特征：评论是否包含内容安全风险表达
```

最终目标：

```text
训练本地 MacBERT 分类器
-> 对 40803 条全量评论进行五维语义预测
-> 将预测结果与视频源、时间、点赞、回复关系结合
-> 支撑后续内容安全分析和评论区圈层化分析
```

## 2. 目录结构

```text
情感分析/
  README.md

  docs/
    五维语义标签体系与标注规范.md
    4090单标签微调说明.md
    四个单标签分类器微调与全量预测总结.md
    四维语义标签交叉分析方案.md

  config/
    deepseek.example.env
    deepseek.env                    可选，本地 DeepSeek API 配置，不提交

  models/
    base/
      Chinese_macbert_base/          本地 MacBERT 基座模型文件
    finetuned/                       后续建议保存微调模型的位置

  data/
    raw/
      semantic_comments_full.csv     40803 条全量评论底表
    samples/
      semantic_annotation_sample.csv 1500 条抽样评论
      semantic_annotation_review_template.csv
      semantic_sample_summary.md
    training/
      semantic_train.csv             五维伪标签总训练集
      topic.csv                      主题分类任务数据
      stance_target.csv              立场对象分类任务数据
      stance.csv                     立场方向分类任务数据
      emotion.csv                    情绪分类任务数据
      discourse.csv                  话语方式多标签任务数据
      risk.csv                       风险特征多标签任务数据
      semantic_train_summary.md      训练集标签分布摘要

  outputs/
    annotations/
      semantic_annotations_deepseek.jsonl
      semantic_annotations_deepseek.csv
      semantic_annotations_for_review.csv
      semantic_annotations_priority_review.csv
    analysis/
      four_dim_semantic/             四维语义交叉分析报告、图表和表格
    deepseek_raw/                    DeepSeek 每批请求和响应原始记录
    predictions/
      full_topic_predictions.csv
      full_semantic_predictions.csv

  scripts/
    data_prep/
      prepare_semantic_sample.py
      deepseek_semantic_annotate.py
      make_review_priority.py
      build_training_set.py
    finetune/
      train_macbert_single_label.py
      predict_semantic_full.py
    analysis/
      analyze_four_dim_semantics.py
```

## 3. config 文件夹说明

当前 `config/` 中只有：

```text
deepseek.example.env
```

真实的 `deepseek.env` 没有放在这里，因为 API key 不应该提交。之前调用 DeepSeek 时，脚本实际读取的是旧路径：

```text
data/config/deepseek.env
```

`deepseek_semantic_annotate.py` 的读取顺序是：

```text
1. 情感分析/config/deepseek.env
2. data/config/deepseek.env
3. shell 环境变量 DEEPSEEK_API_KEY
```

如果后续需要重新调用 DeepSeek，可以复制模板：

```bash
cp 情感分析/config/deepseek.example.env 情感分析/config/deepseek.env
```

然后填写：

```text
DEEPSEEK_API_KEY=你的 key
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 4. data 文件夹说明

### 4.1 semantic_comments_full.csv

路径：

```text
情感分析/data/raw/semantic_comments_full.csv
```

含义：

```text
从 data/database/xibei_event.db 导出的 40803 条全量评论底表。
这是后续全量预测的输入。
```

字段：

```text
content_id：结构化评论 ID
raw_content_id：B 站原始评论 ID
source_id：视频源 ID
source_title：视频标题
platform_source_id：B 站平台视频 ID
source_url：视频链接
author_user_id：UP 主用户 ID
source_published_at：视频发布时间
content_type：comment 一级评论 / reply 楼中楼回复
user_id：评论用户 ID
user_name：评论用户名
gender：用户公开性别字段
level：用户等级字段
content_text：评论正文
created_at：评论发布时间
like_count：评论点赞数
parent_content_id：父评论 ID
root_content_id：楼中楼根评论 ID
received_reply_count：该评论收到的回复数
raw_file_path：来源原始文件路径
```

### 4.2 semantic_annotation_sample.csv

路径：

```text
情感分析/data/samples/semantic_annotation_sample.csv
```

含义：

```text
从 40803 条全量评论中抽样得到的 1500 条标注样本。
这是 DeepSeek 标注输入。
```

比全量表多两个字段：

```text
sample_id：样本编号，例如 S0001
sample_bucket：抽样来源
```

`sample_bucket` 取值：

```text
top_liked：高赞评论
top_replied：高回复评论
source_stratified：按视频源分层随机
random：全局随机
```

### 4.3 semantic_annotation_review_template.csv

路径：

```text
情感分析/data/samples/semantic_annotation_review_template.csv
```

含义：

```text
人工审核空模板。
当前已决定不人工审核，所以这个文件只是保留备用。
```

它在样本字段后面增加了空白标注字段：

```text
topic_label
stance_target
stance_label
emotion_label
discourse_labels
risk_labels
intensity
confidence
need_review
annotation_reason
```

## 5. data/training 文件夹说明

当前训练集来自 DeepSeek 自动标注结果。因为排除了 `need_review=true` 的样本，所以可用训练样本数是：

```text
1349 条
```

### 5.1 semantic_train.csv

路径：

```text
情感分析/data/training/semantic_train.csv
```

含义：

```text
五维语义伪标签总训练集。
```

字段：

```text
sample_id：样本编号
sample_bucket：抽样来源
content_id：评论 ID
source_id：视频源 ID
source_title：视频标题
content_type：comment / reply
user_id：用户 ID
user_name：用户名
content_text：评论正文
created_at：评论发布时间
like_count：点赞数
received_reply_count：收到回复数
topic_label：主题标签
stance_target：立场对象
stance_label：立场方向
emotion_label：情绪标签
discourse_labels：话语方式标签，可能多标签，用分号分隔
risk_labels：风险特征标签，可能多标签，用分号分隔
intensity：表达强度，1-5
confidence：DeepSeek 标注置信度
need_review：是否需要复核
annotation_reason：标注理由
split：train / valid / test
```

用途：

```text
总表适合检查数据、做整体统计、以后做多任务模型。
不建议第一步直接用总表训练一个大模型。
```

### 5.2 单标签任务文件

以下文件用于先训练普通分类器：

```text
topic.csv
stance_target.csv
stance.csv
emotion.csv
```

共同字段：

```text
content_id：评论 ID
content_text：评论正文
标签字段：不同文件不同
split：train / valid / test
```

具体含义：

```text
topic.csv
  标签字段：topic_label
  任务：主题分类

stance_target.csv
  标签字段：stance_target
  任务：判断评论主要针对谁

stance.csv
  标签字段：stance_label
  任务：判断评论支持、质疑、中立还是不明确

emotion.csv
  标签字段：emotion_label
  任务：情绪分类
```

这些任务是单标签多分类：

```text
一条评论只对应一个主标签
使用 softmax + CrossEntropyLoss
```

### 5.3 多标签任务文件

以下文件用于后续训练多标签分类器：

```text
discourse.csv
risk.csv
```

字段：

```text
content_id
content_text
discourse_labels 或 risk_labels
split
```

任务含义：

```text
discourse.csv
  任务：话语方式分类
  标签示例：meme;sarcasm

risk.csv
  任务：内容安全风险特征分类
  标签示例：reputational_attack;emotional_amplification
```

这些任务是多标签分类：

```text
一条评论可能有多个标签
使用 sigmoid + BCEWithLogitsLoss
```

注意：

```text
risk.csv 当前极度类别不平衡。
low_risk_discussion 占绝大多数，其他风险标签样本很少。
风险分类器可以训练，但效果大概率不稳定。
建议最后训练，并在报告中说明风险标签样本不足。
```

### 5.4 semantic_train_summary.md

路径：

```text
情感分析/data/training/semantic_train_summary.md
```

含义：

```text
训练集标签分布摘要。
```

后续训练前必须先看它，确认类别是否严重失衡。

当前一个重要观察：

```text
risk_labels 中 low_risk_discussion 占绝大多数。
DeepSeek 对风险特征标注偏保守。
```

## 6. outputs 文件夹说明

### 6.1 semantic_annotations_deepseek.jsonl

含义：

```text
DeepSeek 标注结果的 JSONL 版。
每行是一条评论的结构化标注。
```

用途：

```text
适合程序读取和追溯原始结构化标签。
```

### 6.2 semantic_annotations_deepseek.csv

含义：

```text
DeepSeek 标注结果的 CSV 版，共 1500 条。
```

用途：

```text
这是 AI 原始标注结果，不建议直接手改。
```

### 6.3 semantic_annotations_for_review.csv

含义：

```text
原本用于人工审核的文件，共 1500 条。
```

当前状态：

```text
已决定不人工审核。
因此这个文件实际就是 DeepSeek 标注后的工作版。
build_training_set.py 从它生成训练集。
```

### 6.4 semantic_annotations_priority_review.csv

含义：

```text
优先人工审核清单，共 433 条。
```

当前状态：

```text
已决定不人工审核。
该文件保留备用，可作为后续检查标签质量的抽样文件。
```

字段 `review_priority_reason` 表示为什么这条评论值得优先看：

```text
need_review
low_confidence
risk_label
top_liked
top_replied
random_check
```

### 6.5 deepseek_raw/

含义：

```text
DeepSeek 每批请求和响应原始记录。
```

文件：

```text
batch_0001.json
...
batch_0150.json
```

每批 10 条评论。

用途：

```text
如果后续发现某条标签异常，可以回溯当时发给 DeepSeek 的 prompt 和返回内容。
```

## 7. scripts 文件夹说明

### 7.1 prepare_semantic_sample.py

作用：

```text
从 data/database/xibei_event.db 读取 40803 条评论
导出 semantic_comments_full.csv
抽样 1500 条生成 semantic_annotation_sample.csv
生成空白审核模板 semantic_annotation_review_template.csv
生成抽样摘要 semantic_sample_summary.md
```

运行：

```bash
python3 情感分析/scripts/data_prep/prepare_semantic_sample.py --sample-size 1500
```

当前已经运行过，一般不需要重复。

### 7.2 deepseek_semantic_annotate.py

作用：

```text
读取 semantic_annotation_sample.csv
按批调用 DeepSeek API
输出五维语义标签
保存 JSONL、CSV、审核文件和原始 batch 响应
```

运行：

```bash
python3 情感分析/scripts/data_prep/deepseek_semantic_annotate.py --batch-size 10 --resume
```

当前已经完成 1500 条标注，一般不需要重复。

### 7.3 make_review_priority.py

作用：

```text
从 semantic_annotations_for_review.csv 中挑出优先审核样本
生成 semantic_annotations_priority_review.csv
```

运行：

```bash
python3 情感分析/scripts/data_prep/make_review_priority.py
```

当前已生成，后续不人工审核时可不用。

### 7.4 build_training_set.py

作用：

```text
读取 semantic_annotations_for_review.csv
过滤不完整标签
默认排除 need_review=true
划分 train / valid / test
生成 semantic_train.csv
生成各任务训练文件
生成 semantic_train_summary.md
```

运行：

```bash
python3 情感分析/scripts/data_prep/build_training_set.py --input 情感分析/outputs/annotations/semantic_annotations_for_review.csv
```

当前已经运行过，生成了 1349 条训练样本。

## 8. 本地模型文件

本地 MacBERT 基座模型已经移动到：

```text
情感分析/models/base/Chinese_macbert_base/
```

其中主要文件是：

```text
情感分析/models/base/Chinese_macbert_base/chinese_macbert_base/
  chinese_macbert_base.ckpt.data-00000-of-00001
  chinese_macbert_base.ckpt.index
  chinese_macbert_base.ckpt.meta
  macbert_base_config.json
  vocab.txt
```

重要注意：

```text
这个本地模型看起来是 TensorFlow checkpoint 格式，
不是 Hugging Face 已转换好的 PyTorch 模型目录。
```

因此，后续训练脚本不能盲目假设：

```python
AutoModel.from_pretrained("情感分析/models/base/Chinese_macbert_base")
```

一定可用。

下一步需要先做模型加载验证：

```text
方案 A：使用 transformers 从 hfl/chinese-macbert-base 下载/加载模型
方案 B：将本地 TensorFlow checkpoint 转换为 Hugging Face 格式
方案 C：尝试用 from_tf=True 读取本地 checkpoint
```

如果网络不可用，优先考虑方案 B 或 C。

## 9. 下一步要做什么

下一步不是重新标注，也不是重新抽样，而是开始微调。

建议新增以下脚本：

```text
情感分析/scripts/train_macbert_single_label.py
情感分析/scripts/train_macbert_multi_label.py
情感分析/scripts/predict_semantic_full.py
```

### 9.1 第一步：训练单标签分类器

先训练 4 个稳定任务：

```text
emotion       读取 data/training/emotion.csv
topic         读取 data/training/topic.csv
stance_target 读取 data/training/stance_target.csv
stance        读取 data/training/stance.csv
```

推荐输出目录：

```text
情感分析/models/finetuned/emotion/
情感分析/models/finetuned/topic/
情感分析/models/finetuned/stance_target/
情感分析/models/finetuned/stance/
```

单标签分类训练方式：

```text
输入：content_text
输出：一个标签
损失函数：CrossEntropyLoss
评估指标：accuracy、macro-F1、每类 precision/recall/F1
```

建议先从 `emotion.csv` 开始，因为最容易解释。

### 9.2 第二步：训练多标签分类器

再训练：

```text
discourse 读取 data/training/discourse.csv
risk      读取 data/training/risk.csv
```

推荐输出目录：

```text
情感分析/models/finetuned/discourse/
情感分析/models/finetuned/risk/
```

多标签分类训练方式：

```text
输入：content_text
输出：多个标签
损失函数：BCEWithLogitsLoss
评估指标：micro-F1、macro-F1、每标签 precision/recall/F1
```

注意：

```text
risk 任务类别极度不平衡，应该最后做。
如果模型几乎全部预测 low_risk_discussion，不要惊讶。
```

### 9.3 第三步：全量预测

训练好模型后，对全量 40803 条评论预测：

```text
输入：data/raw/semantic_comments_full.csv
输出：outputs/predictions/full_semantic_predictions.csv
```

预测结果至少应包含：

```text
content_id
content_text
source_id
source_title
content_type
created_at
like_count
received_reply_count
pred_topic_label
pred_stance_target
pred_stance_label
pred_emotion_label
pred_discourse_labels
pred_risk_labels
```

### 9.4 第四步：交叉分析

全量预测后，进入真正的课程分析：

```text
评论语义 × 视频源
评论语义 × 时间
评论语义 × 点赞
评论语义 × 回复关系
评论语义 × 评论层级
```

这一步服务最终主题：

```text
复杂评论语义如何在争议事件中被放大、聚集，并形成局部圈层化或内容安全风险。
```

## 10. 推荐给下一个 AI 的执行顺序

如果新开对话，请直接从这里继续：

```text
1. 阅读本 README
2. 检查 情感分析/data/training/semantic_train_summary.md
3. 检查本地模型 情感分析/models/base/Chinese_macbert_base/ 是否能被 transformers 加载
4. 如果不能直接加载，先写转换或加载适配逻辑
5. 编写 train_macbert_single_label.py
6. 先用 emotion.csv 跑通一个单标签分类器
7. 保存模型到 情感分析/models/finetuned/emotion/
8. 查看 valid/test 指标
9. 用同一脚本训练 topic、stance_target、stance
10. 再写 train_macbert_multi_label.py 训练 discourse 和 risk
11. 最后写 predict_semantic_full.py 对 40803 条评论全量预测
```

不要重复做：

```text
不要重新抽样
不要重新调用 DeepSeek 标注 1500 条
不要默认需要人工审核
```

除非明确发现数据或标签有严重问题。

## 11. 当前最重要的注意点

```text
1. 当前训练集是 DeepSeek 伪标签，不是人工金标准。
2. 先训练单标签任务，不要一开始就做复杂多任务模型。
3. risk 任务类别极度不平衡，最后处理。
4. 本地 MacBERT 是 TensorFlow checkpoint 格式，训练前必须验证加载方式。
5. 微调输出都应放在 情感分析/models/finetuned/ 下。
6. 全量预测结果应保留原评论元数据，方便后续做交叉分析。
```

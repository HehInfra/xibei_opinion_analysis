# xibei_opinion_analysis

课程项目中的社交媒体评论数据处理与探索性分析原型。

本项目以“罗永浩西贝事件”相关 B 站视频评论区为例。`data/raw/` 中的原始数据由其他组员提供，我负责将这些 raw 文本数据解析、清洗并组织成可复用的结构化数据和 SQLite 数据库。

基于数据库继续做的主题、情绪、立场、用户互动网络和评论区圈层化分析，是我目前个人提出的一版探索性分析思路，尚未和小组成员充分讨论确认，不代表最终的小组方案。后续可以根据课程汇报重点和组内讨论继续调整。

## 已完成的 raw 数据处理

```text
raw 评论数据
  -> interim 中间解析与审计
  -> layer1 第一层事实数据表
  -> SQLite 数据库
```

处理脚本是 `data/process_bilibili_raw.py`。整体思路是先从原始文本中识别不同视频来源、评论内容、回复内容、用户信息和评论互动关系，再把这些信息拆成相对稳定的事实表，最后写入数据库，作为后续分析的统一数据底座。

处理结果主要分为三层：

- `data/interim/`：中间解析结果和审计文件，用于检查 raw 数据是否被正确读取、拆分和清洗。包括 `raw_records.csv`、`raw_file_audit.csv`、`bilibili_video_meta.csv` 以及清洗后的 JSON 文件。
- `data/layer1/`：第一层事实数据表，是后续入库和分析的基础。`sources.csv` 保存视频/内容源，`users.csv` 保存用户，`contents.csv` 保存评论和回复文本，`relations.csv` 保存评论回复关系和用户互动关系。
- `data/database/xibei_event.db`：SQLite 数据库，承接 `layer1` 的结构化结果，方便后续通过 SQL 做关联查询、统计分析和模块化处理。

其中，`layer1/*.csv` 和 `xibei_event.db` 是同一套结构化结果的两种形态：CSV 更方便人工检查和共享，数据库更方便后续程序分析。

## 个人探索性分析思路

在 raw 数据已经结构化入库之后，我基于数据库暂时设计了六个分析模块。这部分是我个人目前的处理思路，主要用于课程项目的原型验证和后续讨论，不是已经由全组共同确认的最终分析框架。

六个模块分别是：

1. 数据概览：统计数据规模、来源分布、时间分布、用户参与情况。
2. 热度分析：观察不同视频、评论、用户和时间段的热度差异。
3. 文本主题分析：用规则初标和 DeepSeek 校正识别评论讨论主题。
4. 情绪与立场分析：识别评论情绪倾向以及对西贝、罗永浩等对象的立场。
5. 用户互动网络分析：基于回复关系观察用户互动结构、关键用户和立场互动。
6. 评论区圈层化分析：尝试从主题集中度、情绪一致性、立场一致性和互动同质性等角度描述评论区的信息茧房或圈层化特征。

对应文件：

- 分析脚本：`data/scripts/analysis_module*.py`
- 分析结果：`data/analysis/`
- 可视化生成脚本：`visualization/build_visualization.py`
- 可视化页面：`visualization/index.html`

## 目录结构

```text
data/
  raw/                         原始 B 站评论数据
  interim/                     中间解析结果与审计报告
  layer1/                      sources/users/contents/relations 第一层事实表
  database/xibei_event.db      SQLite 数据库
  analysis/                    六模块分析结果
  scripts/                     六模块分析脚本
  config/deepseek.example.env  DeepSeek 配置模板
  process_bilibili_raw.py      raw 数据解析脚本

visualization/
  build_visualization.py       可视化页面生成脚本
  index.html                   自包含可视化页面
```

## 运行方式

重新解析 raw 数据：

```bash
python3 data/process_bilibili_raw.py
```

依次运行六个分析模块：

```bash
python3 data/scripts/analysis_module1_overview.py
python3 data/scripts/analysis_module2_heat.py
python3 data/scripts/analysis_module3_topic.py --reuse-llm-raw
python3 data/scripts/analysis_module4_sentiment_stance.py --reuse-llm-raw
python3 data/scripts/analysis_module5_network.py
python3 data/scripts/analysis_module6_cocoon.py
```

如需重新调用 DeepSeek API，而不是复用已保存的 `llm_raw/` 结果：

```bash
cp data/config/deepseek.example.env data/config/deepseek.env
# 编辑 data/config/deepseek.env，填入自己的 DEEPSEEK_API_KEY
python3 data/scripts/analysis_module3_topic.py --use-llm
python3 data/scripts/analysis_module4_sentiment_stance.py --use-llm
```

也可以直接通过环境变量配置：

```bash
DEEPSEEK_API_KEY=your_key python3 data/scripts/analysis_module3_topic.py --use-llm
```

生成可视化页面：

```bash
python3 visualization/build_visualization.py
```

然后打开：

```text
visualization/index.html
```

## 方法说明

模块 3 和模块 4 当前采用：

```text
规则初标 + DeepSeek 大模型校正
```

DeepSeek API 的配置模板位于 `data/config/deepseek.example.env`。实际使用时可复制为 `data/config/deepseek.env` 并填入自己的密钥；该本地配置文件不会提交到 Git。

人工校验暂未执行，因此当前分析结果适合课程设计原型展示、趋势观察和方法讨论，不应作为高精度人工标注数据集使用。

## 课程设计定位

本项目分析的是 B 站视频评论区中的局部观点聚集、情绪表达与互动结构，不直接代表全网舆情传播网络。

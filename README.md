# xibei_opinion_analysis

面向事件型信息茧房识别的社交媒体评论数据工程与分析原型。

本项目以“罗永浩西贝事件”相关 B 站视频评论区为例，完成从 raw 评论数据解析、结构化存储，到主题、情绪、立场、用户互动网络和评论区圈层化指标分析的端到端流程。

## 项目主线

```text
raw 评论数据
  -> 第一层事实数据建模
  -> SQLite / CSV 存储
  -> 六模块分析
  -> 可视化展示
```

六个分析模块：

```text
1. 数据概览
2. 热度分析
3. 文本主题分析
4. 情绪与立场分析
5. 用户互动网络分析
6. 评论区圈层化分析
```

## 目录结构

```text
data/
  raw/                         原始 B 站评论数据
  interim/                     中间解析结果与审计报告
  layer1/                      sources/users/contents/relations 第一层事实表
  database/xibei_event.db      SQLite 数据库
  analysis/                    六模块分析结果
  analysis_module*.py          分析脚本
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
python3 data/analysis_module1_overview.py
python3 data/analysis_module2_heat.py
python3 data/analysis_module3_topic.py --reuse-llm-raw --model deepseek-v4-flash
python3 data/analysis_module4_sentiment_stance.py --reuse-llm-raw --model deepseek-v4-flash
python3 data/analysis_module5_network.py
python3 data/analysis_module6_cocoon.py
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

人工校验暂未执行，因此分析结果适合课程设计原型展示和趋势判断，不应作为高精度人工标注数据集使用。

## 课程设计定位

本项目分析的是 B 站视频评论区中的局部观点聚集、情绪表达与互动结构，不直接代表全网舆情传播网络。


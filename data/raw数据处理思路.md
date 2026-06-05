# raw 数据处理思路

## 1. 当前 raw 数据状态

当前 `data/raw/` 目录中保存的是围绕西贝事件收集的 B 站视频评论数据。

这些文件虽然后缀是 `.txt`，但大部分内容并不是普通文本，而是由两部分组成：

```text
第 1 段：B 站评论接口返回的 JSON 数据
第 2 段：人工或脚本追加的视频元数据块
```

评论 JSON 中主要包含：

```text
cursor：分页信息、评论总数、next_offset
replies：当前页评论列表
member：评论用户信息
content：评论文本
ctime：评论时间
like：点赞数
root / parent / replies：评论层级与楼中楼回复
```

尾部追加的视频元数据块通常包含：

```text
video_name
video_time
video_user_uid
```

需要注意的是，尾部元数据块不一定是严格合法 JSON，因此不能直接把整个 `.txt` 文件作为 JSON 解析。

## 2. 处理总原则

`data/raw/` 中的原始数据只作为原始证据归档，不直接修改、不直接分析。

总体处理流程是：

```text
data/raw/
    ↓
原始文件体检
    ↓
解包评论 JSON 与视频元数据
    ↓
生成第一层事实数据
    ↓
写入 CSV 或 SQLite 数据库
    ↓
进入后续文本分析、关系分析和可视化
```

一句话概括：

> raw 不动，先审计解包，再结构化成 layer1，最后入库分析。

## 3. 目录建议

建议后续将数据目录组织为：

```text
data/
  raw/
    原始 txt 文件
  interim/
    raw_file_audit.csv
    bilibili_clean_json/
    bilibili_video_meta.csv
  layer1/
    sources.csv
    users.csv
    contents.csv
    relations.csv
  database/
    xibei_event.db
```

目录说明：

```text
raw/：保留原始文件，不直接改动
interim/：保存解包、清洗、审计后的中间结果
layer1/：保存第一层事实数据表
database/：保存 SQLite 数据库
```

## 4. 第一步：原始文件体检

在正式解析前，需要先对 `data/raw/` 中每个文件做体检。

建议输出：

```text
data/interim/raw_file_audit.csv
```

体检字段包括：

```text
raw_file_path
file_name
file_size
is_empty
can_extract_comment_json
has_video_meta
video_name
video_time
video_user_uid
comment_all_count
page_reply_count
has_next_offset
nested_reply_count
error_message
```

体检目标：

```text
识别空文件
识别不能解析的文件
确认每个文件是否包含评论 JSON
确认每个文件是否包含视频元数据
统计每个视频当前页评论数和总评论数
判断是否还有可继续分页采集的 next_offset
```

## 5. 第二步：解包 raw txt

因为 raw 文件不是严格 JSON，所以需要先解包。

每个 raw 文件应拆成：

```text
评论 JSON
视频元数据
原始文件路径
```

处理方式：

```text
1. 从文件开头使用 JSON 解析器读取第一个完整 JSON 对象
2. 将第一个 JSON 对象之后的文本视为尾部元数据
3. 从尾部元数据中提取 video_name、video_time、video_user_uid
4. 如果尾部元数据格式不合法，可以使用规则或正则提取
5. 原始文件不修改，只把解包结果写入 interim/
```

解包后的中间结果可以保存为：

```text
data/interim/bilibili_clean_json/{source_id}.json
data/interim/bilibili_video_meta.csv
```

## 6. 第三步：生成第一层事实数据

解包后，将 B 站评论数据归一化为项目第一层数据模型。

第一层事实数据建议包括四张表：

```text
sources.csv
users.csv
contents.csv
relations.csv
```

### 6.1 sources.csv

一个 B 站视频对应一条 source。

建议字段：

```text
source_id
platform
source_type
source_title
platform_source_id
source_url
author_user_id
published_at
raw_file_path
comment_all_count
page_reply_count
has_next_offset
```

字段映射：

```text
platform = bilibili
source_type = video
source_title = video_name 或文件名
platform_source_id = oid / oid_str
author_user_id = video_user_uid
published_at = video_time
raw_file_path = 原始 txt 文件路径
comment_all_count = cursor.all_count
page_reply_count = len(data.replies)
has_next_offset = 是否存在 cursor.pagination_reply.next_offset
```

### 6.2 users.csv

评论用户和回复用户进入 users 表。

建议字段：

```text
user_id
platform
raw_user_id
user_name
user_type
avatar_url
gender
profile_text
level
first_seen_time
last_seen_time
raw_data
```

字段映射：

```text
user_id = bilibili:{member.mid}
platform = bilibili
raw_user_id = member.mid
user_name = member.uname
avatar_url = member.avatar
gender = member.sex
profile_text = member.sign
level = member.level_info.current_level
```

### 6.3 contents.csv

一级评论和楼中楼回复都进入 contents 表。

建议字段：

```text
content_id
platform
source_id
raw_content_id
content_type
user_id
content_text
created_at
like_count
parent_content_id
root_content_id
raw_file_path
raw_data
```

字段映射：

```text
content_id = bilibili_comment:{rpid_str}
platform = bilibili
source_id = bilibili_video:{oid_str}
raw_content_id = rpid_str
content_type = comment 或 reply
user_id = bilibili:{member.mid}
content_text = content.message
created_at = ctime 转换后的时间
like_count = like
parent_content_id = parent / parent_str 对应的评论 ID
root_content_id = root / root_str 对应的评论 ID
```

### 6.4 relations.csv

关系表用于保存评论和回复形成的联系。

建议字段：

```text
relation_id
platform
source_id
source_user_id
target_user_id
relation_type
content_id
target_content_id
created_at
weight
raw_file_path
```

可生成的关系：

```text
一级评论：
用户 comment 视频 source

楼中楼回复：
用户 reply 另一个评论用户

共同参与：
多个用户共同评论同一个视频，可在后续分析层生成 co_participation 弱关系
```

权重建议：

```text
reply = 1.0
comment = 0.5
co_participation = 0.2
```

注意：

```text
一级评论主要是用户对视频的关系，不一定能直接形成用户对用户关系。
楼中楼回复更适合形成用户对用户的强关系。
共同参与关系建议在分析阶段按需要生成，不一定直接写入第一层。
```

## 7. 第四步：写入数据库

这批数据可以先转为 CSV，也可以进一步写入 SQLite。

推荐使用 SQLite，原因是：

```text
不需要部署数据库服务
一个 .db 文件即可保存
方便用 SQL 查询用户、内容和关系
方便后续 Python 分析和可视化读取
```

建议数据库表：

```text
raw_records
sources
users
contents
relations
```

其中 `raw_records` 用于保存原始文件路径、原始 JSON 或原始摘要，保证数据可回溯。

## 8. 后续分析方式

结构化入库后，才能进入分析阶段。

后续可以分析：

```text
评论文本情感
评论立场
评论主题
关键词分布
高赞评论观点
不同视频评论区观点差异
楼中楼回复网络
用户共同参与网络
不同时间阶段的评论变化
```

分析链路如下：

```text
raw txt
    ↓
解析与解包
    ↓
layer1 事实数据
    ↓
SQLite / CSV
    ↓
文本分析
    ↓
关系网络分析
    ↓
可视化展示
```

## 9. 当前数据的使用判断

当前 `data/raw/` 中的数据可以支撑第一层数据入库。

它们已经包含：

```text
评论用户
评论文本
评论时间
点赞数
评论 ID
视频 ID
部分楼中楼回复
评论总数与分页信息
```

但它们还不完全支撑完整舆情网络分析。

当前主要不足：

```text
部分文件可能为空
每个视频大多只包含一页热门评论
缺少完整分页评论
缺少完整视频元数据，例如 BV 号、视频链接、UP 主名称、视频播放量
一级评论主要连接用户与视频，用户之间的直接互动较少
```

因此，当前数据适合先做：

```text
B 站评论区样本分析
评论文本分析
热门评论观点分析
楼中楼回复关系分析
数据入库与可视化原型
```

如果要进一步增强项目效果，后续可以补充：

```text
更多评论分页
更多视频源
微博帖子与评论
新闻评论
抖音 / 小红书评论样本
```


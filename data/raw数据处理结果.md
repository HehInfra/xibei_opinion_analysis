# raw 数据处理结果

本文件记录 `data/raw/` 中 B 站评论 raw 数据的第一轮处理结果。

## 1. 处理脚本

处理脚本：

```text
data/process_bilibili_raw.py
```

脚本执行内容：

```text
1. 扫描 data/raw/ 中的 txt 文件
2. 从每个文件开头解析第一个 B 站评论 JSON
3. 从文件尾部提取 video_name、video_time、video_user_uid
4. 输出 raw 文件审计报告
5. 输出解包后的干净 JSON
6. 生成第一层事实数据 CSV
7. 写入 SQLite 数据库
```

## 2. 输出目录

```text
data/interim/
  raw_file_audit.csv
  bilibili_video_meta.csv
  raw_records.csv
  bilibili_clean_json/

data/layer1/
  sources.csv
  users.csv
  contents.csv
  relations.csv

data/database/
  xibei_event.db
```

## 3. 当前处理统计

第一轮处理结果：

```text
raw 文件数：17
成功解析视频源：16
用户数：434
内容数：455
关系数：455
```

内容类型：

```text
一级评论 comment：312
楼中楼回复 reply：143
```

关系类型：

```text
comment_source：312
reply：143
```

其中：

```text
comment_source 表示用户评论了某个视频
reply 表示用户回复了另一条评论
```

## 4. 发现的问题

有 1 个 raw 文件为空，无法解析：

```text
data/raw/9月16事态升级【罗永浩VS西贝】大卫哥双开巅峰赛！.txt
```

审计报告中记录为：

```text
is_empty = true
can_extract_comment_json = false
error_message = empty_file
```

其余 16 个文件均成功解析出评论 JSON 和视频元数据。

## 5. 已生成的第一层数据

### sources.csv

一个 B 站视频对应一条 source。

主要字段：

```text
source_id
platform
source_type
source_title
platform_source_id
author_user_id
published_at
raw_file_path
comment_all_count
page_reply_count
has_next_offset
```

### users.csv

评论用户和回复用户进入 users 表。

主要字段：

```text
user_id
platform
raw_user_id
user_name
avatar_url
gender
profile_text
level
first_seen_time
last_seen_time
raw_data
```

### contents.csv

一级评论和楼中楼回复进入 contents 表。

主要字段：

```text
content_id
platform
source_id
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

### relations.csv

评论关系和回复关系进入 relations 表。

主要字段：

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

## 6. SQLite 数据库

SQLite 数据库路径：

```text
data/database/xibei_event.db
```

当前数据库表：

```text
raw_records
raw_file_audit
sources
users
contents
relations
```

可以用以下命令查看：

```bash
sqlite3 data/database/xibei_event.db ".tables"
sqlite3 data/database/xibei_event.db "select count(*) from contents;"
```

## 7. 下一步建议

当前数据已经可以进入基础分析。

可以先做：

```text
评论时间分布
各视频评论数量与评论总数对比
高赞评论排行
关键词统计
基础情绪/立场标注
楼中楼回复网络可视化
```

如果要增强项目效果，建议继续补：

```text
1. 重新获取那个空文件对应的视频评论
2. 按 next_offset 补采更多评论页
3. 补充 BV 号、视频链接、UP 主名称、播放量等视频元数据
4. 接入微博或新闻评论数据，形成跨平台对照
```


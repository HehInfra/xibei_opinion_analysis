# 模块 5：用户互动网络分析

## 分析对象

```text
网络类型：楼中楼 reply 强关系网络
节点：参与 reply 关系的 B站用户
边：用户 A 回复 用户 B
```

## 核心统计

```text
节点数：220
reply 边数：143
去重有向边数：143
网络密度：0.002968
弱连通分量数：77
最大连通分量节点数：27
```

## 标签同质性

```text
同立场回复数：72
同立场回复占比：50.35%
跨立场回复数：71
跨立场回复占比：49.65%
同主题回复占比：35.66%
同情绪回复占比：45.45%
```

## 关键用户 Top 10

```text
1. 皇后的头风 | 入度 2 | 出度 13 | 主立场 unclear_meme
2. 飘酱_ | 入度 2 | 出度 1 | 主立场 question_xibei
3. 珠希酱-一分流水 | 入度 2 | 出度 0 | 主立场 support_xibei
4. 末然ン羽 | 入度 2 | 出度 0 | 主立场 question_xibei
5. cvfgji | 入度 2 | 出度 0 | 主立场 question_xibei
6. Mikoyan21 | 入度 2 | 出度 0 | 主立场 question_xibei
7. 橘柚橸 | 入度 2 | 出度 0 | 主立场 unclear_meme
8. 落车six | 入度 2 | 出度 0 | 主立场 unclear_meme
9. MISSC0622 | 入度 2 | 出度 0 | 主立场 question_xibei
10. 弃庸乎 | 入度 2 | 出度 0 | 主立场 unclear_meme
```

## 方法说明

当前模块只将真实 `reply` 关系纳入强关系网络。`comment_source` 表示用户评论视频，不是用户对用户互动，因此不进入主网络。共同评论同一视频的 `co_participation` 弱关系可在后续扩展，但本版不混入强关系结论。

## 输出文件

```text
data/analysis/module5_network/network_overview.json
data/analysis/module5_network/reply_edges.csv
data/analysis/module5_network/user_nodes.csv
data/analysis/module5_network/key_users.csv
data/analysis/module5_network/stance_interaction_matrix.csv
data/analysis/module5_network/network_graph.json
```

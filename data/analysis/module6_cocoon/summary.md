# 模块 6：评论区圈层化分析

## 指标设计

```text
text_cocoon_score =
  0.30 * 主题集中度
+ 0.30 * 情绪一致性
+ 0.40 * 立场一致性

full_cocoon_score =
  0.25 * 主题集中度
+ 0.25 * 情绪一致性
+ 0.30 * 立场一致性
+ 0.20 * 互动同质性
```

其中：

```text
主题集中度 = 视频内最大主题占比
情绪一致性 = 视频内最大情绪占比
立场一致性 = 视频内最大立场占比
互动同质性 = reply 边中同立场互动比例
```

当某视频 reply 边数少于 5 条时，`full_cocoon_score` 标记为结构样本不足，只保留文本版分数。

## 总体结果

```text
视频源数量：16
平均 text_cocoon_score：0.5442
平均 full_cocoon_score：0.5375
文本圈层等级分布：{'medium': 5, 'low': 11}
结构样本状态分布：{'sufficient': 9, 'insufficient': 7}
```

## 圈层化较高的视频

```text
1. 从西门子冰箱到西贝预制菜，罗永浩的“较真”从未缺席家人们！今天必须聊聊罗永浩，当年他怒砸西门子冰箱，为消费者维权，那场面太震撼了！ | text=0.7033(medium) | full=0.7616 | 主立场=unclear_meme(73.33%)
2. 罗永浩：我真没打，他就先捅了自己20多刀...我该怎么办？ | text=0.6333(medium) | full=样本不足 | 主立场=unclear_meme(61.9%)
3. 罗永浩吐槽，西贝戗面馒头每个21元，黑珍珠一钻餐厅包子只要16元 | text=0.6162(medium) | full=0.6095 | 主立场=unclear_meme(64.86%)
4. 罗永浩也是懵了，自杀式公关还是第一次见。 | text=0.6(medium) | full=样本不足 | 主立场=question_xibei(60.0%)
5. 罗永浩：我这辈子没打过这么离谱的仗，对方上来先捅了自己20多刀… | text=0.6(medium) | full=样本不足 | 主立场=unclear_meme(60.0%)
6. 罗永浩与西贝争论事件原委 一条视频告诉你 | text=0.5462(low) | full=样本不足 | 主立场=question_xibei(61.54%)
7. 罗永浩痛批西贝太缺德！一个馒头卖21块，比米其林餐厅还贵！#罗永浩#西贝#预制菜#太离谱 | text=0.524(low) | full=0.538 | 主立场=unclear_meme(56.0%)
8. 罗永浩是怎么把西贝搞翻车的？ | text=0.5333(low) | full=样本不足 | 主立场=question_xibei(76.19%)
```

## 解释边界

当前模块衡量的是 B 站视频评论区中的局部圈层化倾向，不代表全网舆情传播网络。由于模块 3/4 暂未人工校验，且部分视频 reply 边较少，结果适合作为课程设计原型分析和趋势判断。

## 输出文件

```text
data/analysis/module6_cocoon/cocoon_overview.json
data/analysis/module6_cocoon/source_cocoon_scores.csv
data/analysis/module6_cocoon/topic_concentration.csv
data/analysis/module6_cocoon/sentiment_consistency.csv
data/analysis/module6_cocoon/stance_consistency.csv
data/analysis/module6_cocoon/interaction_homophily.csv
data/analysis/module6_cocoon/hot_comment_stance_concentration.csv
```

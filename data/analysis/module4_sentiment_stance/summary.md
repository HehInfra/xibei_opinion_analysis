# 模块 4：情绪与立场分析

## 分析方法

```text
规则初标：已执行
大模型校正：已执行
LLM 响应来源：API 实时调用
人工校验：暂时跳过
```

## 大模型调用

```text
模型：deepseek-v4-flash
成功批次数：23
LLM 应用标签数：455
```

## 情绪分布

```text
neutral: 289 条，占比 63.52%，获赞 45087
sarcastic: 101 条，占比 22.2%，获赞 41125
negative: 57 条，占比 12.53%，获赞 5732
positive: 8 条，占比 1.76%，获赞 8580
```

## 立场分布

```text
unclear_meme: 206 条，占比 45.27%，获赞 40093
question_xibei: 202 条，占比 44.4%，获赞 50950
neutral_discussion: 30 条，占比 6.59%，获赞 840
support_luoyonghao: 8 条，占比 1.76%，获赞 8373
support_xibei: 7 条，占比 1.54%，获赞 155
question_luoyonghao: 2 条，占比 0.44%，获赞 113
```

## 高赞评论标签抽样

```text
10287赞 | sarcastic | question_xibei | 一桌菜只有客人是现宰的[大哭]
5203赞 | neutral | unclear_meme | 一岁的儿童吃着两岁的西兰花[笑哭][笑哭]
4525赞 | positive | support_luoyonghao | 西门子冰箱，西贝事件都说明，这个世界需要一些勇敢的较真的人
3880赞 | sarcastic | unclear_meme | 哈哈哈哈
3547赞 | sarcastic | question_xibei | 我算是西贝的常客，本来一开始并不是很关注，但是直播截图看到转基因大豆油我就绷不住了。姑且不讨论转基因是否对健康有害，这个油它便宜啊。西贝随便人均消费100+，结果就用这40/50一...
3297赞 | positive | support_luoyonghao | 这件事，只要是吃饭馆的，吃外面点餐的，都应该支持老罗。正常人类，理解现做的饭菜，就是我下单后，餐馆把生鲜食材，通过自己厨房餐具完全做熟，端到食客面前。中间不存在半熟，冷冻数月以上的...
3209赞 | sarcastic | question_xibei | 主要是西贝太贵了[doge]你看有人骂萨莉亚吗
2678赞 | sarcastic | unclear_meme | 兄弟们 我昨天去了哈哈哈哈 43元的炒鸡蛋[笑哭]
2521赞 | neutral | question_xibei | [笑哭]当西贝厨房整了去年的羊腿的时候，什么预制菜定义权都不重要了
2318赞 | sarcastic | unclear_meme | 最可怜的是会有人说：谁一大早给你买那么多鱼来做菜啊[doge]，彻底绷不住了
```

## 质量提示

```text
需要复核数量：272
```

当前结果保存在：

```text
data/analysis/module4_sentiment_stance/content_sentiment_stance.csv
data/analysis/module4_sentiment_stance/sentiment_summary_by_source.csv
data/analysis/module4_sentiment_stance/stance_summary_by_source.csv
data/analysis/module4_sentiment_stance/sentiment_stance_overview.json
```

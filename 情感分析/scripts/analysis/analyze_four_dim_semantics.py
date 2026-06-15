from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

SEMANTIC_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("XDG_CACHE_HOME", str(SEMANTIC_DIR / "outputs" / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(SEMANTIC_DIR / "outputs" / ".cache" / "matplotlib"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = SEMANTIC_DIR / "outputs" / "predictions" / "full_semantic_predictions.csv"
DEFAULT_OUTPUT_DIR = SEMANTIC_DIR / "outputs" / "analysis" / "four_dim_semantic"

TOPIC_ORDER = [
    "prepared_food_authenticity",
    "meme_offtopic",
    "price_value",
    "brand_pr_crisis",
    "other_unclear",
    "luoyonghao_persona",
    "consumer_right_to_know",
    "industry_trust",
    "food_safety_child_meal",
]
EMOTION_ORDER = ["neutral", "sarcastic", "negative", "positive", "disappointed", "angry", "anxious"]
STANCE_TARGET_ORDER = ["xibei", "unclear", "industry", "luoyonghao", "consumers"]
STANCE_ORDER = ["question", "neutral", "support", "unclear"]
LIKES_ORDER = ["0", "1-9", "10-99", "100-999", "1000+"]
REPLIES_ORDER = ["0", "1-9", "10-49", "50+"]


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    df["date"] = df["created_at_dt"].dt.date.astype(str)
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["received_reply_count"] = pd.to_numeric(df["received_reply_count"], errors="coerce").fillna(0).astype(int)
    df["like_bucket"] = pd.cut(
        df["like_count"],
        bins=[-1, 0, 9, 99, 999, math.inf],
        labels=LIKES_ORDER,
    ).astype(str)
    df["reply_bucket"] = pd.cut(
        df["received_reply_count"],
        bins=[-1, 0, 9, 49, math.inf],
        labels=REPLIES_ORDER,
    ).astype(str)
    df["is_high_like_top5"] = df["like_count"] >= df["like_count"].quantile(0.95)
    df["is_high_like_top1"] = df["like_count"] >= df["like_count"].quantile(0.99)
    return df


def distribution(df: pd.DataFrame, column: str, order: list[str] | None = None) -> pd.DataFrame:
    result = (
        df[column]
        .value_counts()
        .rename_axis(column)
        .reset_index(name="comments")
    )
    if order:
        order_map = {label: idx for idx, label in enumerate(order)}
        result["_order"] = result[column].map(order_map).fillna(999)
        result = result.sort_values(["_order", "comments"], ascending=[True, False]).drop(columns="_order")
    result["share"] = result["comments"] / len(df)
    return result


def crosstab_count_share(df: pd.DataFrame, row: str, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = pd.crosstab(df[row], df[col])
    share = count.div(count.sum(axis=1), axis=0).fillna(0)
    return count, share


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["source_id", "source_title"], dropna=False)
    rows = []
    for (source_id, title), group in grouped:
        topic_dist = group["pred_topic_label"].value_counts(normalize=True)
        emotion_dist = group["pred_emotion_label"].value_counts(normalize=True)
        stance_dist = group["pred_stance_label"].value_counts(normalize=True)
        target_dist = group["pred_stance_target_label"].value_counts(normalize=True)
        rows.append(
            {
                "source_id": source_id,
                "source_title": title,
                "comments": len(group),
                "top_topic": topic_dist.index[0],
                "top_topic_share": topic_dist.iloc[0],
                "sarcastic_share": emotion_dist.get("sarcastic", 0.0),
                "question_share": stance_dist.get("question", 0.0),
                "xibei_target_share": target_dist.get("xibei", 0.0),
                "avg_like_count": group["like_count"].mean(),
                "avg_received_reply_count": group["received_reply_count"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("comments", ascending=False)


def high_like_lift(df: pd.DataFrame, column: str) -> pd.DataFrame:
    overall = df[column].value_counts(normalize=True)
    top5 = df.loc[df["is_high_like_top5"], column].value_counts(normalize=True)
    top1 = df.loc[df["is_high_like_top1"], column].value_counts(normalize=True)
    labels = sorted(set(overall.index) | set(top5.index) | set(top1.index))
    rows = []
    for label in labels:
        overall_share = overall.get(label, 0.0)
        top5_share = top5.get(label, 0.0)
        top1_share = top1.get(label, 0.0)
        rows.append(
            {
                column: label,
                "overall_share": overall_share,
                "top5_share": top5_share,
                "top1_share": top1_share,
                "top5_lift": top5_share / overall_share if overall_share else None,
                "top1_lift": top1_share / overall_share if overall_share else None,
            }
        )
    return pd.DataFrame(rows).sort_values("top5_lift", ascending=False, na_position="last")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True if df.index.name else False, encoding="utf-8-sig")


def setup_plot() -> None:
    preferred_fonts = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_name = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")
    sns.set_theme(style="whitegrid", font=font_name)
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, x_label: str = "Share") -> None:
    plt.figure(figsize=(9, 5.4))
    plot_df = df.copy()
    if "share" in plot_df.columns:
        plot_df["share_pct"] = plot_df["share"] * 100
        value_col = "share_pct"
        x_label = "Share of comments (%)"
    sns.barplot(data=plot_df, y=label_col, x=value_col, color="#4477aa")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_heatmap(df: pd.DataFrame, title: str, path: Path, fmt: str = ".0%") -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap="Blues", annot=True, fmt=fmt, linewidths=0.5, cbar=True)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_line(df: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(11, 5.5))
    for col in df.columns:
        plt.plot(df.index, df[col] * 100, marker="o", linewidth=1.8, markersize=4, label=col)
    plt.title(title)
    plt.ylabel("Share of comments (%)")
    plt.xlabel("Date")
    tick_step = max(1, math.ceil(len(df.index) / 18))
    tick_positions = list(range(0, len(df.index), tick_step))
    plt.xticks(tick_positions, [df.index[index] for index in tick_positions], rotation=35, ha="right")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def short_text(value: str, max_len: int = 28) -> str:
    text = str(value)
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def topn_sentence(dist: pd.DataFrame, label_col: str, n: int = 3) -> str:
    parts = []
    for row in dist.head(n).itertuples(index=False):
        label = getattr(row, label_col)
        comments = getattr(row, "comments")
        share = getattr(row, "share")
        parts.append(f"{label} {comments:,} 条（{fmt_pct(share)}）")
    return "，".join(parts)


def write_html_report(
    *,
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, str],
    output_path: Path,
) -> None:
    topic_dist = tables["topic_distribution"]
    emotion_dist = tables["emotion_distribution"]
    stance_target_dist = tables["stance_target_distribution"]
    stance_dist = tables["stance_distribution"]
    source = tables["source_summary"]
    high_like_topic = tables["high_like_topic_lift"]
    high_like_emotion = tables["high_like_emotion_lift"]
    content_type_stance = tables["content_type_stance_share"]

    top_source = source.iloc[0]
    top_like_topic = high_like_topic.iloc[0]
    top_like_emotion = high_like_emotion.iloc[0]
    reply_count = int((df["content_type"] == "reply").sum())
    comment_count = int((df["content_type"] == "comment").sum())

    def img(key: str, alt: str) -> str:
        return f'<figure><img src="figures/{figures[key]}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>西贝事件 B 站评论四维语义交叉分析</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; line-height: 1.65; margin: 0; background: #f6f7f9; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 40px 28px 64px; background: #fff; }}
    h1 {{ font-size: 30px; margin: 0 0 16px; }}
    h2 {{ font-size: 22px; margin-top: 40px; border-top: 1px solid #d9dee7; padding-top: 24px; }}
    h3 {{ font-size: 17px; margin-top: 28px; }}
    .meta {{ color: #5c6675; font-size: 14px; }}
    .summary {{ background: #eef5ff; border-left: 4px solid #4477aa; padding: 16px 20px; margin: 20px 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 12px; background: #fbfcfe; }}
    .card strong {{ display: block; font-size: 22px; color: #243b53; }}
    figure {{ margin: 22px 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #e3e8ef; border-radius: 6px; }}
    figcaption {{ color: #626f80; font-size: 13px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f8; }}
    code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>西贝事件 B 站评论四维语义交叉分析</h1>
  <p class="meta">基于 <code>full_semantic_predictions.csv</code> 的 {len(df):,} 条评论；四维标签来自 MacBERT 单标签分类器预测。</p>

  <h2>Executive Summary</h2>
  <div class="summary">
    <p><strong>讨论主轴已经从单一事实判断扩展为多议题舆论结构。</strong> 主题前三类为 {topn_sentence(topic_dist, "pred_topic_label")}；预制菜真实性、玩梗跑题和价格价值共同构成评论区主体。</p>
    <p><strong>评论区不是单纯愤怒型，而是反讽和中性表达并存。</strong> 情绪前三类为 {topn_sentence(emotion_dist, "pred_emotion_label")}；反讽表达占比高，是本事件传播风格的重要特征。</p>
    <p><strong>质疑表达占主导，且对象主要指向西贝。</strong> 立场方向中 question 为 {fmt_pct(float(stance_dist.loc[stance_dist["pred_stance_label"] == "question", "share"].iloc[0]))}，立场对象中 xibei 为 {fmt_pct(float(stance_target_dist.loc[stance_target_dist["pred_stance_target_label"] == "xibei", "share"].iloc[0]))}。</p>
    <p><strong>点赞机制更容易放大具备传播性的表达。</strong> Top 5% 高赞评论中，{top_like_topic["pred_topic_label"]} 的相对占比提升最高；情绪上 {top_like_emotion["pred_emotion_label"]} 的高赞提升最明显。</p>
  </div>

  <div class="cards">
    <div class="card"><span>全量评论</span><strong>{len(df):,}</strong></div>
    <div class="card"><span>视频源</span><strong>{df["source_id"].nunique():,}</strong></div>
    <div class="card"><span>一级评论</span><strong>{comment_count:,}</strong></div>
    <div class="card"><span>楼中楼回复</span><strong>{reply_count:,}</strong></div>
  </div>

  <h2>整体舆论结构：真实性争议、玩梗和价格价值共同主导</h2>
  <p><strong>主题分布显示，事件讨论并不只围绕“是否预制菜”。</strong> 预制菜真实性是最大主题，但玩梗跑题几乎同样庞大，说明评论区已经进入事件传播后的娱乐化和反讽化表达阶段；价格价值和品牌公关危机也形成稳定讨论板块。</p>
  {img("topic_distribution", "主题分布：预制菜真实性与玩梗跑题构成最大两类")}
  <p><strong>情绪结构更接近“中性围观 + 反讽传播”，而不是直接愤怒。</strong> 中性评论数量最高，反讽紧随其后；负面表达存在但不是唯一主轴。情绪小类样本和模型表现较弱，angry/anxious/disappointed 不适合做强解释。</p>
  {img("emotion_distribution", "情绪分布：neutral 与 sarcastic 占据主导")}
  <p><strong>立场对象和立场方向交叉后，最核心的组合是“质疑西贝”。</strong> 同时，unclear 和 neutral 的体量也很大，说明大量评论是玩梗、围观、信息复述或对象不明确的讨论。</p>
  {img("stance_target_stance_heatmap", "立场对象 × 立场方向：质疑主要集中指向西贝")}

  <h2>视频源差异：不同视频承载不同评论区氛围</h2>
  <p><strong>评论最多的视频并不只是样本量大，也在塑造可见舆论场。</strong> 评论量最高的视频是“{top_source["source_title"]}”，共有 {int(top_source["comments"]):,} 条评论；其 top topic 是 {top_source["top_topic"]}，最大主题占比 {fmt_pct(float(top_source["top_topic_share"]))}。</p>
  {img("source_topic_heatmap", "Top 12 视频源 × 主题占比：不同视频评论区呈现不同主题氛围")}
  <p>该图适合用于区分“事实争议型视频”“品牌危机型视频”和“玩梗围观型视频”。后续写报告时，可以围绕评论量最高的视频和主题集中度最高的视频做案例解释。</p>

  <h2>点赞放大：高赞评论更可能凸显传播性表达</h2>
  <p><strong>点赞不是简单反映全部评论，而是放大更容易被认同和转述的表达。</strong> 按点赞层级看，主题结构会发生变化；高赞区更适合观察平台内部可见性的放大机制。</p>
  {img("like_bucket_topic_share", "点赞层级 × 主题占比：高赞区的主题结构与总体分布不同")}
  <p><strong>情绪维度上，反讽和负面表达是否被放大，是内容安全解释的关键。</strong> 如果高赞层级中 sarcastic 或 negative 占比上升，可以解释为平台互动机制强化了更具传播性的情绪表达。</p>
  {img("like_bucket_emotion_share", "点赞层级 × 情绪占比：高赞评论中的情绪结构")}

  <h2>评论层级：楼中楼回复是争论与附和的重要场域</h2>
  <p><strong>本数据中楼中楼回复多于一级评论。</strong> 一级评论 {comment_count:,} 条，楼中楼回复 {reply_count:,} 条。评论层级差异可以帮助区分“首轮观点表达”和“互动中的争论/附和”。</p>
  {img("content_type_stance_share", "评论层级 × 立场方向：一级评论与楼中楼回复的立场结构")}
  <p>如果楼中楼回复中 question 或 sarcastic 更高，说明互动区更可能承担争论、反驳和情绪共振功能；如果一级评论中某些主题更集中，则说明它们更像事件入口处的直接观点表达。</p>

  <h2>时间演化：议题迁移需要结合事件节点解释</h2>
  <p><strong>按日观察主题占比，可以判断事件讨论是否发生迁移。</strong> 这里先展示主要主题随日期变化的形态；后续如果要进入课程报告，可以把关键视频发布时间、品牌回应节点或直播节点叠加到时间线上。</p>
  {img("daily_topic_share", "每日主要主题占比变化：观察讨论是否从事实争议迁移到公关、价格和玩梗")}

  <h2>Recommended Next Steps</h2>
  <ol>
    <li><strong>先把本轮表格和图表用于报告主干。</strong> 重点写整体语义结构、视频源差异、点赞放大和评论层级差异。</li>
    <li><strong>抽查高置信与低置信样本。</strong> 每个主要标签抽 20 条，确认模型预测是否能支撑文字结论。</li>
    <li><strong>补充事件节点时间线。</strong> 将品牌回应、罗永浩相关视频发布时间、争议爆发节点叠加到日期图中。</li>
    <li><strong>暂缓风险模型作为主线。</strong> 当前可先用“反讽 + 质疑 + 点赞 + 回复”解释内容安全风险表达的放大机制。</li>
  </ol>

  <h2>Caveats and Assumptions</h2>
  <ul>
    <li>四维标签是模型预测结果，训练标签来自 DeepSeek 伪标签，不是人工金标准。</li>
    <li>小类标签需要谨慎解释，尤其是 food_safety_child_meal、industry_trust、angry、anxious、disappointed、consumers。</li>
    <li>点赞数代表平台内部互动反馈和可见性，不等于真实公众态度。</li>
    <li>用户公开资料不应被过度解释为真实人口属性。</li>
  </ul>
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze four-dimensional semantic predictions.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dirs = ensure_dirs(output_dir)
    setup_plot()

    df = read_data(Path(args.input))
    tables: dict[str, pd.DataFrame] = {}

    tables["topic_distribution"] = distribution(df, "pred_topic_label", TOPIC_ORDER)
    tables["emotion_distribution"] = distribution(df, "pred_emotion_label", EMOTION_ORDER)
    tables["stance_target_distribution"] = distribution(df, "pred_stance_target_label", STANCE_TARGET_ORDER)
    tables["stance_distribution"] = distribution(df, "pred_stance_label", STANCE_ORDER)

    count, share = crosstab_count_share(df, "pred_topic_label", "pred_emotion_label")
    tables["topic_emotion_count"] = count
    tables["topic_emotion_share"] = share

    count, share = crosstab_count_share(df, "pred_topic_label", "pred_stance_label")
    tables["topic_stance_count"] = count
    tables["topic_stance_share"] = share

    count, share = crosstab_count_share(df, "pred_stance_target_label", "pred_stance_label")
    tables["stance_target_stance_count"] = count
    tables["stance_target_stance_share"] = share

    tables["source_summary"] = source_summary(df)

    top_sources = tables["source_summary"].head(12)["source_id"].tolist()
    source_chart_df = df.loc[df["source_id"].isin(top_sources)].copy()
    source_chart_df["source_short_title"] = source_chart_df["source_title"].map(short_text)
    source_topic = pd.crosstab(
        source_chart_df["source_short_title"],
        source_chart_df["pred_topic_label"],
        normalize="index",
    )
    source_topic = source_topic[[col for col in TOPIC_ORDER if col in source_topic.columns]]
    tables["top_source_topic_share"] = source_topic

    like_topic = pd.crosstab(df["like_bucket"], df["pred_topic_label"], normalize="index").reindex(LIKES_ORDER)
    like_topic = like_topic[[col for col in TOPIC_ORDER if col in like_topic.columns]]
    tables["like_bucket_topic_share"] = like_topic

    like_emotion = pd.crosstab(df["like_bucket"], df["pred_emotion_label"], normalize="index").reindex(LIKES_ORDER)
    like_emotion = like_emotion[[col for col in EMOTION_ORDER if col in like_emotion.columns]]
    tables["like_bucket_emotion_share"] = like_emotion

    content_type_stance = pd.crosstab(df["content_type"], df["pred_stance_label"], normalize="index")
    content_type_stance = content_type_stance[[col for col in STANCE_ORDER if col in content_type_stance.columns]]
    tables["content_type_stance_share"] = content_type_stance

    reply_topic = pd.crosstab(df["reply_bucket"], df["pred_topic_label"], normalize="index").reindex(REPLIES_ORDER)
    reply_topic = reply_topic[[col for col in TOPIC_ORDER if col in reply_topic.columns]]
    tables["reply_bucket_topic_share"] = reply_topic

    daily_counts = df["date"].value_counts().sort_index().rename_axis("date").reset_index(name="comments")
    tables["daily_comment_count"] = daily_counts
    active_dates = daily_counts.loc[daily_counts["comments"] >= 30, "date"]
    daily_topic = pd.crosstab(df.loc[df["date"].isin(active_dates), "date"], df.loc[df["date"].isin(active_dates), "pred_topic_label"], normalize="index")
    daily_topic = daily_topic[[col for col in TOPIC_ORDER[:6] if col in daily_topic.columns]]
    tables["daily_topic_share"] = daily_topic

    tables["high_like_topic_lift"] = high_like_lift(df, "pred_topic_label")
    tables["high_like_emotion_lift"] = high_like_lift(df, "pred_emotion_label")
    tables["high_like_stance_lift"] = high_like_lift(df, "pred_stance_label")

    for name, table in tables.items():
        save_csv(table, dirs["tables"] / f"{name}.csv")

    figures: dict[str, str] = {
        "topic_distribution": "topic_distribution.png",
        "emotion_distribution": "emotion_distribution.png",
        "stance_target_stance_heatmap": "stance_target_stance_heatmap.png",
        "source_topic_heatmap": "source_topic_heatmap.png",
        "like_bucket_topic_share": "like_bucket_topic_share.png",
        "like_bucket_emotion_share": "like_bucket_emotion_share.png",
        "content_type_stance_share": "content_type_stance_share.png",
        "daily_topic_share": "daily_topic_share.png",
    }

    save_bar(tables["topic_distribution"], "pred_topic_label", "share", "Topic distribution", dirs["figures"] / figures["topic_distribution"])
    save_bar(tables["emotion_distribution"], "pred_emotion_label", "share", "Emotion distribution", dirs["figures"] / figures["emotion_distribution"])
    save_heatmap(tables["stance_target_stance_share"].reindex(STANCE_TARGET_ORDER).fillna(0), "Stance target x stance share", dirs["figures"] / figures["stance_target_stance_heatmap"])
    save_heatmap(tables["top_source_topic_share"].fillna(0), "Top sources x topic share", dirs["figures"] / figures["source_topic_heatmap"], fmt=".0%")
    save_heatmap(tables["like_bucket_topic_share"].fillna(0), "Like bucket x topic share", dirs["figures"] / figures["like_bucket_topic_share"], fmt=".0%")
    save_heatmap(tables["like_bucket_emotion_share"].fillna(0), "Like bucket x emotion share", dirs["figures"] / figures["like_bucket_emotion_share"], fmt=".0%")
    save_heatmap(tables["content_type_stance_share"].fillna(0), "Content type x stance share", dirs["figures"] / figures["content_type_stance_share"], fmt=".0%")
    save_line(tables["daily_topic_share"].fillna(0), "Daily topic share", dirs["figures"] / figures["daily_topic_share"])

    metrics = {
        "rows": int(len(df)),
        "sources": int(df["source_id"].nunique()),
        "comments": int((df["content_type"] == "comment").sum()),
        "replies": int((df["content_type"] == "reply").sum()),
        "avg_confidence": {
            "topic": float(df["pred_topic_confidence"].mean()),
            "emotion": float(df["pred_emotion_confidence"].mean()),
            "stance_target": float(df["pred_stance_target_confidence"].mean()),
            "stance": float(df["pred_stance_confidence"].mean()),
        },
    }
    (dirs["base"] / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_html_report(
        df=df,
        tables=tables,
        figures=figures,
        output_path=dirs["base"] / "report.html",
    )

    print(f"Analysis complete: {dirs['base']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

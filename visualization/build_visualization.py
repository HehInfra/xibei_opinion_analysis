from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
OUT = ROOT / "visualization" / "index.html"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def short(text: str, limit: int = 48) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def build_payload() -> dict[str, Any]:
    module1 = read_json(ANALYSIS / "module1_overview" / "overview.json")
    module2 = read_json(ANALYSIS / "module2_heat" / "heat_overview.json")
    module3 = read_json(ANALYSIS / "module3_topic" / "topic_overview.json")
    module4 = read_json(ANALYSIS / "module4_sentiment_stance" / "sentiment_stance_overview.json")
    module5 = read_json(ANALYSIS / "module5_network" / "network_overview.json")
    module6 = read_json(ANALYSIS / "module6_cocoon" / "cocoon_overview.json")

    video_heat = read_csv(ANALYSIS / "module2_heat" / "video_heat.csv")
    hot_comments = read_csv(ANALYSIS / "module2_heat" / "hot_comments_top100.csv")
    content_labels = read_csv(ANALYSIS / "module4_sentiment_stance" / "content_sentiment_stance.csv")
    network_graph = read_json(ANALYSIS / "module5_network" / "network_graph.json")
    cocoon_scores = read_csv(ANALYSIS / "module6_cocoon" / "source_cocoon_scores.csv")
    daily_heat = read_csv(ANALYSIS / "module2_heat" / "daily_heat.csv")

    topic_summary = [
        {
            "label": label,
            "count": data["content_count"],
            "percent": data["content_percent"],
            "likes": data["like_sum"],
        }
        for label, data in module3["topic_summary"].items()
    ]
    topic_summary.sort(key=lambda item: item["count"], reverse=True)

    sentiment_summary = [
        {"label": label, "count": data["content_count"], "percent": data["content_percent"], "likes": data["like_sum"]}
        for label, data in module4["sentiment_summary"].items()
    ]
    sentiment_summary.sort(key=lambda item: item["count"], reverse=True)

    stance_summary = [
        {"label": label, "count": data["content_count"], "percent": data["content_percent"], "likes": data["like_sum"]}
        for label, data in module4["stance_summary"].items()
    ]
    stance_summary.sort(key=lambda item: item["count"], reverse=True)

    topic_labels = sorted({row["topic_label"] for row in content_labels})
    stance_labels = sorted({row["stance_label"] for row in content_labels})
    heat_counter: Counter[tuple[str, str]] = Counter()
    for row in content_labels:
        heat_counter[(row["topic_label"], row["stance_label"])] += 1
    topic_stance_heatmap = [
        {"topic": topic, "stance": stance, "count": heat_counter[(topic, stance)]}
        for topic in topic_labels
        for stance in stance_labels
    ]

    top_videos = [
        {
            "title": short(row["source_title"], 36),
            "full_title": row["source_title"],
            "likes": to_int(row["extracted_like_sum"]),
            "platform_comments": to_int(row["platform_comment_count"]),
            "sample_count": to_int(row["extracted_content_count"]),
            "heat_score": to_float(row["heat_score"]),
        }
        for row in video_heat[:10]
    ]

    top_comments = [
        {
            "rank": row["rank"],
            "user": row["user_name"],
            "likes": to_int(row["like_count"]),
            "text": row["content_text"],
            "source": short(row["source_title"], 32),
        }
        for row in hot_comments[:10]
    ]

    label_by_content = {row["content_id"]: row for row in content_labels}
    for item in top_comments:
        # hot comment ids are already in hot_comments, add labels by text fallback if needed
        pass
    top_comments_labeled = []
    for row in hot_comments[:10]:
        label_row = label_by_content.get(row["content_id"], {})
        top_comments_labeled.append(
            {
                "rank": row["rank"],
                "user": row["user_name"],
                "likes": to_int(row["like_count"]),
                "text": row["content_text"],
                "source": short(row["source_title"], 32),
                "topic": label_row.get("topic_label", ""),
                "sentiment": label_row.get("sentiment_label", ""),
                "stance": label_row.get("stance_label", ""),
            }
        )

    nodes = sorted(network_graph["nodes"], key=lambda row: int(row.get("total_degree") or 0), reverse=True)[:80]
    keep = {node["id"] for node in nodes}
    links = [
        link
        for link in network_graph["links"]
        if link["source"] in keep and link["target"] in keep
    ][:160]

    cocoon_rank = [
        {
            "title": short(row["source_title"], 40),
            "full_title": row["source_title"],
            "text_score": to_float(row["text_cocoon_score"]),
            "full_score": None if row["full_cocoon_score"] == "" else to_float(row["full_cocoon_score"]),
            "level": row["text_cocoon_level"],
            "dominant_stance": row["dominant_stance"],
            "stance_consistency": to_float(row["stance_consistency"]),
            "structure": row["structure_sample_status"],
        }
        for row in cocoon_scores
    ]

    return {
        "overview": {
            "counts": module1["counts"],
            "time_range": module1["time_range"],
            "heat_summary": module2["summary"],
            "network": {
                "node_count": module5["node_count"],
                "reply_edge_count": module5["reply_edge_count"],
                "density": module5["density"],
                "component_count": module5["component_count"],
                "same_stance_reply_percent": module5["same_stance_reply_percent"],
            },
            "cocoon": {
                "average_text_cocoon_score": module6["average_text_cocoon_score"],
                "average_full_cocoon_score": module6["average_full_cocoon_score"],
                "text_level_distribution": module6["text_level_distribution"],
                "structure_sample_distribution": module6["structure_sample_distribution"],
            },
        },
        "daily_heat": daily_heat,
        "top_videos": top_videos,
        "top_comments": top_comments_labeled,
        "topic_summary": topic_summary,
        "sentiment_summary": sentiment_summary,
        "stance_summary": stance_summary,
        "topic_labels": topic_labels,
        "stance_labels": stance_labels,
        "topic_stance_heatmap": topic_stance_heatmap,
        "network": {"nodes": nodes, "links": links},
        "cocoon_rank": cocoon_rank,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>罗永浩西贝事件评论区分析可视化</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --blue: #2f6fdd;
      --teal: #139b8f;
      --red: #d14b4b;
      --yellow: #d99722;
      --green: #3c8b4f;
      --purple: #7a5cc8;
      --pink: #c55383;
      --cyan: #2589a8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
      letter-spacing: 0;
    }
    header {
      background: #17202a;
      color: #fff;
      padding: 22px 28px 18px;
      border-bottom: 4px solid #d99722;
    }
    h1 { margin: 0; font-size: 26px; line-height: 1.25; font-weight: 760; }
    .subtitle { margin-top: 8px; color: #cbd5e1; font-size: 14px; line-height: 1.6; max-width: 1100px; }
    main { padding: 20px; max-width: 1500px; margin: 0 auto; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 0 0 8px; font-size: 14px; color: #344054; }
    .grid { display: grid; gap: 12px; }
    .cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
      background: #fbfcfe;
    }
    .metric .label { color: var(--muted); font-size: 13px; }
    .metric .value { font-size: 26px; font-weight: 760; margin-top: 8px; }
    .metric .note { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .chart { width: 100%; min-height: 260px; }
    .chart.tall { min-height: 440px; }
    .bar-row { display: grid; grid-template-columns: minmax(120px, 260px) 1fr 70px; gap: 10px; align-items: center; margin: 8px 0; }
    .bar-label { font-size: 12px; color: #344054; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { height: 18px; background: #eef2f6; border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--blue); border-radius: 3px; }
    .bar-value { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 8px 0 2px; font-size: 12px; color: var(--muted); }
    .swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 5px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #edf0f5; padding: 8px; text-align: left; vertical-align: top; }
    th { color: #475467; background: #f8fafc; font-weight: 650; }
    .tag { display: inline-block; padding: 2px 6px; border-radius: 4px; background: #eef2f6; color: #344054; white-space: nowrap; }
    .muted { color: var(--muted); }
    .heatmap { display: grid; gap: 2px; overflow-x: auto; padding-bottom: 4px; }
    .heat-cell { min-width: 78px; height: 42px; display: flex; align-items: center; justify-content: center; color: #17202a; font-size: 12px; border-radius: 3px; }
    .axis-cell { min-width: 78px; font-size: 11px; color: #475467; display: flex; align-items: center; justify-content: center; padding: 4px; text-align: center; }
    svg { width: 100%; height: auto; display: block; }
    .footnote { color: var(--muted); font-size: 12px; line-height: 1.6; margin-top: 10px; }
    @media (max-width: 980px) {
      main { padding: 12px; }
      .cols-4, .cols-3, .cols-2 { grid-template-columns: 1fr; }
      header { padding: 18px; }
      h1 { font-size: 21px; }
      .bar-row { grid-template-columns: 130px 1fr 56px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>罗永浩西贝事件 B站评论区六模块分析</h1>
    <div class="subtitle">从 raw 评论数据出发，经过结构化存储、热度分析、主题识别、情绪立场判断、互动网络构建，最终形成评论区局部圈层化指标。</div>
  </header>
  <main>
    <section>
      <h2>1. 数据概览</h2>
      <div class="grid cols-4" id="metricCards"></div>
      <div class="grid cols-2" style="margin-top:12px">
        <div>
          <h3>评论/回复时间分布</h3>
          <div id="dailyHeat"></div>
        </div>
        <div>
          <h3>分析链路</h3>
          <div class="footnote">
            raw txt/JSON → sources/users/contents/relations → SQLite → 六模块分析 → 可视化展示。当前结果适合课程设计原型分析；模块 3/4 暂未进行人工校验，相关样本保留 need_review 字段。
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>2. 热度分析</h2>
      <div class="grid cols-2">
        <div>
          <h3>视频样本获赞排行</h3>
          <div id="videoHeat"></div>
        </div>
        <div>
          <h3>高赞评论 Top 10</h3>
          <div id="hotComments"></div>
        </div>
      </div>
    </section>

    <section>
      <h2>3. 文本主题分析</h2>
      <div class="grid cols-2">
        <div>
          <h3>主题分布</h3>
          <div id="topicBars"></div>
        </div>
        <div>
          <h3>主题说明</h3>
          <div class="footnote">
            主题标签由规则初标和 DeepSeek 校正得到。占比最高的主题集中在预制菜与食材真实性、罗永浩个人形象、价格与性价比，以及调侃玩梗。
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>4. 情绪与立场分析</h2>
      <div class="grid cols-2">
        <div>
          <h3>情绪分布</h3>
          <div id="sentimentBars"></div>
        </div>
        <div>
          <h3>立场分布</h3>
          <div id="stanceBars"></div>
        </div>
      </div>
    </section>

    <section>
      <h2>5. 主题-立场交叉热力图</h2>
      <div id="topicStanceHeatmap"></div>
      <div class="footnote">颜色越深表示该主题和立场组合下的评论数量越多。</div>
    </section>

    <section>
      <h2>6. 用户回复网络</h2>
      <div class="grid cols-3">
        <div class="metric"><div class="label">reply 网络节点</div><div class="value" id="networkNodeCount"></div><div class="note">参与楼中楼回复的用户</div></div>
        <div class="metric"><div class="label">reply 边</div><div class="value" id="networkEdgeCount"></div><div class="note">用户对用户回复</div></div>
        <div class="metric"><div class="label">同立场回复占比</div><div class="value" id="sameStancePercent"></div><div class="note">基于模块 4 标签</div></div>
      </div>
      <div id="networkSvg" style="margin-top:14px"></div>
      <div class="footnote">节点颜色表示用户主立场；线条表示 reply 关系。当前强关系网络较稀疏，因此主要用于展示局部互动结构。</div>
    </section>

    <section>
      <h2>7. 评论区圈层化指标</h2>
      <div class="grid cols-2">
        <div>
          <h3>CocoonScore 排行</h3>
          <div id="cocoonBars"></div>
        </div>
        <div>
          <h3>指标解释</h3>
          <div class="footnote">
            text_cocoon_score = 主题集中度、情绪一致性、立场一致性的加权结果。full_cocoon_score 在此基础上加入 reply 同立场互动比例；当 reply 边不足时标记为结构样本不足。
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const DATA = __DATA__;
    const colors = {
      question_xibei: '#d14b4b',
      support_xibei: '#3c8b4f',
      support_luoyonghao: '#2f6fdd',
      question_luoyonghao: '#7a5cc8',
      neutral_discussion: '#139b8f',
      unclear_meme: '#d99722',
      neutral: '#139b8f',
      sarcastic: '#d99722',
      negative: '#d14b4b',
      positive: '#3c8b4f'
    };

    function fmt(n) {
      if (n === null || n === undefined || n === '') return '';
      const num = Number(n);
      if (!Number.isFinite(num)) return String(n);
      return num.toLocaleString('zh-CN');
    }

    function esc(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function barList(el, items, options = {}) {
      const max = Math.max(...items.map(d => Number(d.value) || 0), 1);
      const color = options.color || '#2f6fdd';
      el.innerHTML = items.map(d => {
        const width = Math.max(2, (Number(d.value) || 0) / max * 100);
        return `<div class="bar-row" title="${esc(d.title || d.label)}">
          <div class="bar-label">${esc(d.label)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${d.color || color}"></div></div>
          <div class="bar-value">${fmt(d.value)}</div>
        </div>`;
      }).join('');
    }

    function renderMetrics() {
      const c = DATA.overview.counts;
      const h = DATA.overview.heat_summary;
      const co = DATA.overview.cocoon;
      const cards = [
        ['视频源', c.sources, '成功解析的 B站视频源'],
        ['用户', c.users, '参与评论或回复的用户'],
        ['评论/回复', c.contents, '结构化内容总量'],
        ['样本获赞', h.total_likes, '当前样本点赞总量'],
        ['reply 关系', DATA.overview.network.reply_edge_count, '用户对用户强关系'],
        ['平均文本圈层分', co.average_text_cocoon_score, '基于主题/情绪/立场'],
        ['同立场回复', DATA.overview.network.same_stance_reply_percent + '%', 'reply 边中的同立场比例'],
        ['结构样本充足', co.structure_sample_distribution.sufficient || 0, 'reply 边数达到阈值的视频']
      ];
      document.getElementById('metricCards').innerHTML = cards.map(([label, value, note]) =>
        `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${fmt(value)}</div><div class="note">${esc(note)}</div></div>`
      ).join('');
    }

    function renderDailyHeat() {
      barList(document.getElementById('dailyHeat'), DATA.daily_heat.map(d => ({
        label: d.date,
        value: d.content_count,
        title: `${d.date}: ${d.content_count}`
      })), {color:'#139b8f'});
    }

    function renderHeat() {
      barList(document.getElementById('videoHeat'), DATA.top_videos.map(d => ({
        label: d.title,
        value: d.likes,
        title: d.full_title
      })), {color:'#2f6fdd'});
      document.getElementById('hotComments').innerHTML = `<table><thead><tr><th>#</th><th>评论</th><th>赞</th><th>主题</th><th>立场</th></tr></thead><tbody>` +
        DATA.top_comments.map(d => `<tr>
          <td>${d.rank}</td><td>${esc(d.text)}</td><td>${fmt(d.likes)}</td><td><span class="tag">${esc(d.topic)}</span></td><td><span class="tag">${esc(d.stance)}</span></td>
        </tr>`).join('') + `</tbody></table>`;
    }

    function renderTopics() {
      barList(document.getElementById('topicBars'), DATA.topic_summary.map(d => ({
        label: d.label,
        value: d.count,
        color: '#7a5cc8'
      })));
      barList(document.getElementById('sentimentBars'), DATA.sentiment_summary.map(d => ({
        label: d.label,
        value: d.count,
        color: colors[d.label] || '#2f6fdd'
      })));
      barList(document.getElementById('stanceBars'), DATA.stance_summary.map(d => ({
        label: d.label,
        value: d.count,
        color: colors[d.label] || '#2f6fdd'
      })));
    }

    function renderHeatmap() {
      const topics = DATA.topic_labels;
      const stances = DATA.stance_labels;
      const map = new Map(DATA.topic_stance_heatmap.map(d => [`${d.topic}|${d.stance}`, d.count]));
      const max = Math.max(...DATA.topic_stance_heatmap.map(d => d.count), 1);
      let html = `<div class="heatmap" style="grid-template-columns:160px repeat(${stances.length}, 88px)">`;
      html += `<div></div>` + stances.map(s => `<div class="axis-cell">${esc(s)}</div>`).join('');
      for (const topic of topics) {
        html += `<div class="axis-cell" style="justify-content:flex-start">${esc(topic)}</div>`;
        for (const stance of stances) {
          const value = map.get(`${topic}|${stance}`) || 0;
          const alpha = value / max;
          const bg = `rgba(47,111,221,${0.08 + alpha * 0.78})`;
          html += `<div class="heat-cell" style="background:${bg}" title="${esc(topic)} × ${esc(stance)}: ${value}">${value || ''}</div>`;
        }
      }
      html += `</div>`;
      document.getElementById('topicStanceHeatmap').innerHTML = html;
    }

    function renderNetwork() {
      const net = DATA.overview.network;
      document.getElementById('networkNodeCount').textContent = fmt(net.node_count);
      document.getElementById('networkEdgeCount').textContent = fmt(net.reply_edge_count);
      document.getElementById('sameStancePercent').textContent = net.same_stance_reply_percent + '%';

      const width = 1100, height = 520, cx = width / 2, cy = height / 2;
      const nodes = DATA.network.nodes;
      const links = DATA.network.links;
      const positions = new Map();
      nodes.forEach((n, i) => {
        const r = 70 + (i % 5) * 38 + (Number(n.total_degree) || 0) * 5;
        const angle = i / nodes.length * Math.PI * 2;
        positions.set(n.id, {x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r});
      });
      const maxDegree = Math.max(...nodes.map(n => Number(n.total_degree) || 0), 1);
      const linkSvg = links.map(l => {
        const s = positions.get(l.source), t = positions.get(l.target);
        if (!s || !t) return '';
        const stroke = l.same_dominant_stance === 'true' ? '#139b8f' : '#98a2b3';
        return `<line x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" stroke="${stroke}" stroke-width="${1 + Number(l.weight || 1) * 0.5}" opacity="0.45" />`;
      }).join('');
      const nodeSvg = nodes.map(n => {
        const p = positions.get(n.id);
        const radius = 4 + (Number(n.total_degree) || 0) / maxDegree * 12;
        const fill = colors[n.dominant_stance] || '#667085';
        return `<g><circle cx="${p.x}" cy="${p.y}" r="${radius}" fill="${fill}" opacity="0.9"><title>${esc(n.name || n.id)} | ${esc(n.dominant_stance)} | degree ${n.total_degree}</title></circle></g>`;
      }).join('');
      document.getElementById('networkSvg').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="用户回复网络">${linkSvg}${nodeSvg}</svg>
        <div class="legend">${Object.entries(colors).filter(([k]) => DATA.stance_labels.includes(k)).map(([k,v]) => `<span><i class="swatch" style="background:${v}"></i>${k}</span>`).join('')}</div>`;
    }

    function renderCocoon() {
      barList(document.getElementById('cocoonBars'), DATA.cocoon_rank.slice(0, 12).map(d => ({
        label: d.title,
        value: Math.round(d.text_score * 1000) / 1000,
        title: d.full_title,
        color: d.level === 'medium' ? '#d99722' : '#139b8f'
      })), {color:'#d99722'});
    }

    renderMetrics();
    renderDailyHeat();
    renderHeat();
    renderTopics();
    renderHeatmap();
    renderNetwork();
    renderCocoon();
  </script>
</body>
</html>
"""


def main() -> int:
    payload = build_payload()
    html_text = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

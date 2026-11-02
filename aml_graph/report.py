"""Write pipeline results to disk: CSVs, a Markdown summary, GraphML and an HTML network view."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from .pipeline import Result, suspicious_subgraph

TYPOLOGY_COLORS = {
    "fan_in": "#e8590c",
    "fan_out": "#f59f00",
    "cycle": "#c2255c",
    "layering": "#7048e8",
    "structuring": "#1c7ed6",
    "": "#868e96",
}

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AML Suspicious Network</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root { --bg:#f8f9fa; --fg:#212529; --panel:#ffffff; --muted:#6c757d; --border:#dee2e6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --fg:#e9ecef; --panel:#212429; --muted:#adb5bd; --border:#343a40; }
  }
  body { margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:12px 16px; border-bottom:1px solid var(--border); }
  h1 { font-size:18px; margin:0 0 4px; }
  .legend span { display:inline-block; margin-right:14px; font-size:13px; color:var(--muted); }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
  #wrap { display:flex; height:calc(100vh - 70px); }
  #net { flex:1; }
  #info { width:320px; padding:12px 16px; border-left:1px solid var(--border); background:var(--panel);
          overflow:auto; font-size:13px; }
  #info td { padding:2px 8px 2px 0; vertical-align:top; }
  @media (max-width: 700px) { #wrap { flex-direction:column; } #info { width:auto; border-left:0; } }
</style>
</head>
<body>
<header>
  <h1>Suspicious transaction network</h1>
  <div class="legend">__LEGEND__</div>
</header>
<div id="wrap"><div id="net"></div><div id="info">Click an account to see its details. Node size = risk score; edge width = amount.</div></div>
<script>
const data = __DATA__;
const nodes = new vis.DataSet(data.nodes);
const edges = new vis.DataSet(data.edges);
const net = new vis.Network(document.getElementById('net'), {}, {
  layout: { improvedLayout: false },
  physics: { stabilization: { iterations: 400 }, barnesHut: { springLength: 120 } },
  edges: { arrows: { to: { enabled: true, scaleFactor: 0.5 } }, smooth: { type: 'continuous' },
           color: { color: '#adb5bd', highlight: '#495057' } },
  nodes: { shape: 'dot', scaling: { min: 6, max: 22 },
           font: { size: 11, color: getComputedStyle(document.body).color } },
  interaction: { hover: true, tooltipDelay: 120 }
});
// Freeze the layout once settled: separate schemes would otherwise keep drifting apart.
// vis restores its own camera after this event, so fit on the next tick.
net.once('stabilizationIterationsDone', () => {
  net.setOptions({ physics: false });
  setTimeout(() => net.fit(), 0);
});
// Load data only after layout so the canvas has its real size when fitting.
window.addEventListener('load', () => requestAnimationFrame(() => net.setData({ nodes, edges })));
net.on('click', p => {
  if (!p.nodes.length) return;
  const n = data.nodes.find(x => x.id === p.nodes[0]);
  const rows = Object.entries(n.meta).map(([k, v]) => `<tr><td><b>${k}</b></td><td>${v}</td></tr>`).join('');
  document.getElementById('info').innerHTML = `<h3>${n.id}</h3><table>${rows}</table>`;
});
</script>
</body>
</html>
"""


def _network_html(sub: nx.DiGraph, scores: pd.DataFrame) -> str:
    max_amt = max((d["amount"] for _, _, d in sub.edges(data=True)), default=1.0)
    nodes = []
    for n in sub.nodes:
        s = scores.loc[n]
        typ = s["typologies"].split(",")[0] if s["typologies"] else ""
        nodes.append(
            {
                "id": n,
                "label": n,
                "value": float(s["risk_score"]) * 10 + 1,
                "color": TYPOLOGY_COLORS.get(typ, TYPOLOGY_COLORS[""]),
                "borderWidth": 3 if s["flagged"] else 1,
                "title": f"{n} - risk {s['risk_score']:.2f}",
                "meta": {
                    "kind": sub.nodes[n].get("kind", ""),
                    "risk score": f"{s['risk_score']:.3f}",
                    "detector score": f"{s['detector_score']:.3f}",
                    "association score": f"{s['association_score']:.3f}",
                    "typologies": s["typologies"] or "-",
                    "total in": f"{s['total_in']:,.0f}",
                    "total out": f"{s['total_out']:,.0f}",
                    "pass-through": f"{s['pass_through']:.2f}",
                    "alerts": s["alert_ids"] or "-",
                },
            }
        )
    edges = [
        {
            "from": u,
            "to": v,
            "width": 1 + 6 * d["amount"] / max_amt,
            "title": f"{d['count']} tx, {d['amount']:,.0f}",
        }
        for u, v, d in sub.edges(data=True)
    ]
    legend = "".join(
        f'<span><i style="background:{c}"></i>{t or "association only"}</span>' for t, c in TYPOLOGY_COLORS.items()
    )
    return HTML_TEMPLATE.replace("__LEGEND__", legend).replace(
        "__DATA__", json.dumps({"nodes": nodes, "edges": edges})
    )


def _markdown(result: Result) -> str:
    lines = ["# AML Graph Analysis Report", ""]
    s = result.scores
    lines += [
        f"- Accounts analysed: **{len(s)}**",
        f"- Alerts raised: **{len(result.alerts)}**",
        f"- Accounts flagged: **{int(s['flagged'].sum())}**",
        "",
        "## Alerts by typology",
        "",
        "| Typology | Alerts | Mean score |",
        "|---|---:|---:|",
    ]
    for typ, grp in result.alerts.groupby("typology"):
        lines.append(f"| {typ} | {len(grp)} | {grp['score'].mean():.2f} |")

    m = result.metrics
    if m:
        lines += [
            "",
            "## Evaluation against ground truth",
            "",
            f"- Account precision **{m['precision']:.1%}**, recall **{m['recall']:.1%}**, F1 **{m['f1']:.3f}**",
            f"- Precision@k (k = {m['laundering_accounts']} true laundering accounts): **{m['precision_at_k']:.1%}**",
            f"- Alert precision (alerts mostly made of laundering transactions): **{m['alert_precision']:.1%}**",
            "",
            "| Typology | Planted patterns | Caught | Recall |",
            "|---|---:|---:|---:|",
        ]
        for typ, st in sorted(m["pattern_recall"].items()):
            lines.append(f"| {typ} | {st['patterns']} | {st['caught']} | {st['recall']:.0%} |")

    lines += ["", "## Top 15 accounts by risk", "", "| Account | Risk | Detector | Association | Typologies |", "|---|---:|---:|---:|---|"]
    for acct, r in s.head(15).iterrows():
        lines.append(
            f"| {acct} | {r['risk_score']:.3f} | {r['detector_score']:.3f} | {r['association_score']:.3f} | {r['typologies'] or '-'} |"
        )

    lines += ["", "## Communities with the most flagged accounts", "", "| Community | Size | Flagged | Share |", "|---:|---:|---:|---:|"]
    for _, r in result.communities.head(10).iterrows():
        lines.append(f"| {r['community']} | {r['size']} | {r['flagged']} | {r['flagged_share']:.1%} |")
    return "\n".join(lines) + "\n"


def write_report(result: Result, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.alerts.to_csv(out / "alerts.csv", index=False)
    result.scores.to_csv(out / "account_scores.csv")
    result.communities.to_csv(out / "communities.csv", index=False)
    if result.metrics:
        (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (out / "report.md").write_text(_markdown(result), encoding="utf-8")

    sub = suspicious_subgraph(result)
    gml = nx.DiGraph()
    for n, d in sub.nodes(data=True):
        gml.add_node(n, kind=d.get("kind", ""), risk=float(result.scores.loc[n, "risk_score"]))
    for u, v, d in sub.edges(data=True):
        gml.add_edge(u, v, amount=d["amount"], count=d["count"])
    nx.write_graphml(gml, out / "suspicious_subgraph.graphml")
    (out / "network.html").write_text(_network_html(sub, result.scores), encoding="utf-8")
    return out

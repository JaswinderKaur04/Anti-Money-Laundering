"""End-to-end pipeline: build graph, run detectors, score accounts, evaluate."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from . import detectors as det
from .graph import TemporalIndex, aggregate_graph, node_features


@dataclass
class PipelineConfig:
    threshold: float = 10_000
    fan_window_days: float = 7
    fan_min_counterparties: int = 8
    structuring_min_count: int = 5
    max_hop_hours: float = 72
    min_chain_hops: int = 4
    alert_score: float = 0.5  # accounts at or above this final score are flagged
    association_weight: float = 0.4  # cap on how much association alone contributes


@dataclass
class Result:
    graph: nx.DiGraph
    transactions: pd.DataFrame
    alerts: pd.DataFrame
    scores: pd.DataFrame
    communities: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run_detectors(tx: pd.DataFrame, accounts: pd.DataFrame, cfg: PipelineConfig) -> list[det.Alert]:
    alerts: list[det.Alert] = []
    for direction in ("in", "out"):
        alerts += det.detect_fan(
            tx, accounts, direction, cfg.fan_window_days, cfg.fan_min_counterparties, cfg.threshold
        )
    alerts += det.detect_structuring(tx, cfg.threshold, min_count=cfg.structuring_min_count)
    alerts += det.detect_cycles_and_layering(
        tx, TemporalIndex.build(tx), cfg.max_hop_hours, min_chain_hops=cfg.min_chain_hops
    )
    alerts.sort(key=lambda a: a.score, reverse=True)
    for i, a in enumerate(alerts):
        a.alert_id = f"AL{i:05d}"
    return alerts


def score_accounts(
    g: nx.DiGraph, alerts: list[det.Alert], feats: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    """Combine alert evidence (noisy-OR) with network association into a final score."""
    survive = pd.Series(1.0, index=feats.index)
    typologies: dict[str, set[str]] = {}
    alert_ids: dict[str, list[str]] = {}
    for a in alerts:
        for acct in a.accounts:
            survive[acct] *= 1 - a.score
            typologies.setdefault(acct, set()).add(a.typology)
            alert_ids.setdefault(acct, []).append(a.alert_id)
    detector = 1 - survive

    seeds = {k: float(v) for k, v in detector.items() if v > 0}
    association = det.propagate_risk(g, seeds).reindex(feats.index).fillna(0.0)
    final = 1 - (1 - detector) * (1 - cfg.association_weight * association)

    out = feats.copy()
    out["detector_score"] = detector.round(4)
    out["association_score"] = association.round(4)
    out["risk_score"] = final.round(4)
    out["flagged"] = out["risk_score"] >= cfg.alert_score
    out["typologies"] = [",".join(sorted(typologies.get(a, ()))) for a in out.index]
    out["alert_ids"] = [",".join(alert_ids.get(a, [])) for a in out.index]
    return out.sort_values("risk_score", ascending=False)


def evaluate(
    accounts: pd.DataFrame, tx: pd.DataFrame, scores: pd.DataFrame, alerts: list[det.Alert]
) -> dict:
    """Compare results with the generator's ground truth labels."""
    truth = accounts.set_index("account_id")["is_laundering"].reindex(scores.index).fillna(False)
    pred = scores["flagged"]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Pattern-level recall: a planted pattern counts as caught if any alert
    # covers at least half of its transactions.
    alert_tx = [set(a.tx_ids) for a in alerts]
    per_typology: dict[str, dict] = {}
    for (typ, _pid), grp in tx[tx["typology"] != ""].groupby(["typology", "pattern_id"]):
        p = set(grp["tx_id"])
        caught = any(len(p & s) >= 0.5 * len(p) for s in alert_tx)
        stats = per_typology.setdefault(typ, {"patterns": 0, "caught": 0})
        stats["patterns"] += 1
        stats["caught"] += int(caught)
    for stats in per_typology.values():
        stats["recall"] = round(stats["caught"] / stats["patterns"], 3)

    bad_tx = set(tx.loc[tx["typology"] != "", "tx_id"])
    true_alerts = sum(len(set(a.tx_ids) & bad_tx) >= 0.5 * len(a.tx_ids) for a in alerts)

    # Ranking quality independent of the threshold: how many bad accounts in the top-k.
    k = int(truth.sum())
    top_k = scores.head(k).index
    return {
        "accounts": int(len(scores)),
        "laundering_accounts": k,
        "flagged_accounts": int(pred.sum()),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "precision_at_k": round(float(truth[top_k].mean()), 3) if k else 0.0,
        "alerts": len(alerts),
        "alert_precision": round(true_alerts / len(alerts), 3) if alerts else 0.0,
        "pattern_recall": per_typology,
    }


def alerts_frame(alerts: list[det.Alert]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alert_id": a.alert_id,
                "typology": a.typology,
                "score": a.score,
                "focal_account": a.focal,
                "n_accounts": len(a.accounts),
                "n_transactions": len(a.tx_ids),
                "details": a.details,
                "accounts": ";".join(a.accounts),
                "tx_ids": ";".join(a.tx_ids),
            }
            for a in alerts
        ],
        columns=[
            "alert_id", "typology", "score", "focal_account", "n_accounts",
            "n_transactions", "details", "accounts", "tx_ids",
        ],
    )


def run(
    tx: pd.DataFrame, accounts: pd.DataFrame, cfg: PipelineConfig | None = None, labelled: bool | None = None
) -> Result:
    cfg = cfg or PipelineConfig()
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    g = aggregate_graph(tx, accounts)
    feats = node_features(tx, g)
    alerts = run_detectors(tx, accounts, cfg)
    scores = score_accounts(g, alerts, feats, cfg)
    communities = det.community_summary(g, set(scores.index[scores["flagged"]]))

    if labelled is None:
        labelled = "is_laundering" in accounts.columns and "typology" in tx.columns
    if "typology" in tx.columns:
        tx["typology"] = tx["typology"].fillna("")
    metrics = evaluate(accounts, tx, scores, alerts) if labelled else {}
    return Result(g, tx, alerts_frame(alerts), scores, communities, metrics)


def suspicious_subgraph(result: Result, max_nodes: int = 400) -> nx.DiGraph:
    """The money flows behind the alerts, limited to the riskiest accounts.

    Only transactions that belong to an alert are drawn, so each scheme shows up
    as its own cluster instead of being buried in ordinary traffic.
    """
    alert_tx = {t for ids in result.alerts["tx_ids"] for t in ids.split(";") if t}
    flows = result.transactions[result.transactions["tx_id"].isin(alert_tx)]
    sub = aggregate_graph(flows)
    for n in sub.nodes:
        sub.nodes[n].update(result.graph.nodes[n])
    ranked = result.scores.loc[result.scores.index.isin(sub.nodes), "risk_score"].sort_values(ascending=False)
    return sub.subgraph(ranked.index[:max_nodes]).copy()


__all__ = ["PipelineConfig", "Result", "run", "suspicious_subgraph"]

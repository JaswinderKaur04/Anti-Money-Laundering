"""Turn a transaction table into graph structures used by the detectors."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


def aggregate_graph(tx: pd.DataFrame, accounts: pd.DataFrame | None = None) -> nx.DiGraph:
    """Collapse parallel transactions into one weighted edge per (src, dst) pair.

    Edge attributes: ``amount`` (total value), ``count`` and ``first``/``last`` timestamps.
    """
    agg = tx.groupby(["src", "dst"]).agg(
        amount=("amount", "sum"),
        count=("amount", "size"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
    g = nx.DiGraph()
    if accounts is not None:
        g.add_nodes_from((r.account_id, {"kind": r.kind}) for r in accounts.itertuples())
    g.add_edges_from(
        (src, dst, {"amount": float(amount), "count": int(count), "first": first, "last": last})
        for (src, dst), amount, count, first, last in agg.itertuples(name=None)
    )
    return g


def node_features(tx: pd.DataFrame, g: nx.DiGraph) -> pd.DataFrame:
    """Per-account flow statistics and centrality, useful context for analysts."""
    out_ = tx.groupby("src")["amount"].agg(total_out="sum", n_out="size")
    in_ = tx.groupby("dst")["amount"].agg(total_in="sum", n_in="size")
    feats = pd.DataFrame(index=pd.Index(list(g.nodes), name="account_id"))
    feats = feats.join(in_).join(out_).fillna(0.0)
    feats["in_degree"] = pd.Series(dict(g.in_degree()))
    feats["out_degree"] = pd.Series(dict(g.out_degree()))
    hi, lo = feats[["total_in", "total_out"]].max(axis=1), feats[["total_in", "total_out"]].min(axis=1)
    # 1.0 means everything that came in went straight back out: a classic mule signature.
    feats["pass_through"] = np.where(hi > 0, lo / hi.where(hi > 0, 1), 0.0)
    feats["pagerank"] = pd.Series(nx.pagerank(g, weight="amount"))
    return feats


@dataclass
class TemporalIndex:
    """Outgoing transactions per account, sorted by time, for fast forward walks."""

    times: dict[str, np.ndarray]
    amounts: dict[str, np.ndarray]
    dsts: dict[str, np.ndarray]
    tx_ids: dict[str, np.ndarray]

    @classmethod
    def build(cls, tx: pd.DataFrame) -> "TemporalIndex":
        times, amounts, dsts, tx_ids = {}, {}, {}, {}
        for src, grp in tx.sort_values("timestamp").groupby("src", sort=False):
            times[src] = grp["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
            amounts[src] = grp["amount"].to_numpy()
            dsts[src] = grp["dst"].to_numpy()
            tx_ids[src] = grp["tx_id"].to_numpy()
        return cls(times, amounts, dsts, tx_ids)

    def successors(self, node: str, after_ns: int, until_ns: int):
        """Yield ``(time_ns, amount, dst, tx_id)`` for transactions out of ``node`` in (after, until]."""
        t = self.times.get(node)
        if t is None:
            return
        lo = np.searchsorted(t, after_ns, side="right")
        hi = np.searchsorted(t, until_ns, side="right")
        for i in range(lo, hi):
            yield int(t[i]), float(self.amounts[node][i]), self.dsts[node][i], self.tx_ids[node][i]

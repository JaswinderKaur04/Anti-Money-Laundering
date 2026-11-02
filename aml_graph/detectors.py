"""Graph-based detectors for common money-laundering typologies.

Each detector returns a list of :class:`Alert` objects. An alert names the
accounts and transactions involved and carries a score in ``[0, 1]``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .graph import TemporalIndex

NS_PER_HOUR = 3_600_000_000_000


@dataclass
class Alert:
    typology: str
    accounts: list[str]
    tx_ids: list[str]
    score: float
    details: str
    focal: str = ""  # the account the alert is centred on, if any
    alert_id: str = field(default="")


def _near_threshold_share(amounts: np.ndarray, threshold: float, band: float) -> float:
    return float(np.mean((amounts >= band * threshold) & (amounts < threshold))) if len(amounts) else 0.0


def _uniformity(amounts: np.ndarray) -> float:
    """1.0 for identical amounts, falling towards 0 as the coefficient of variation grows."""
    if len(amounts) < 2 or amounts.mean() == 0:
        return 0.0
    return float(1.0 / (1.0 + amounts.std() / amounts.mean() * 4))


def detect_fan(
    tx: pd.DataFrame,
    accounts: pd.DataFrame,
    direction: str = "in",
    window_days: float = 7,
    min_counterparties: int = 8,
    threshold: float = 10_000,
    min_amount_share: float = 0.2,
    exclude_kinds: tuple[str, ...] = ("merchant", "business"),
    followup_days: float = 14,
) -> list[Alert]:
    """Find accounts that receive from (``in``) or send to (``out``) many distinct
    counterparties inside a short window: smurfing collectors and dispersal hubs.

    Merchants and businesses legitimately have many counterparties (shoppers,
    payroll), so they are excluded by default; a production system would compare
    against a per-segment baseline instead. Only transfers of at least
    ``min_amount_share * threshold`` count, so everyday card spend is ignored.
    For fan-in, large onward transfers within ``followup_days`` are added to the
    alert so the cash-out account is captured too.
    """
    focal_col, other_col = ("dst", "src") if direction == "in" else ("src", "dst")
    excluded = set(accounts.loc[accounts["kind"].isin(exclude_kinds), "account_id"])
    window = pd.Timedelta(days=window_days)
    tx = tx[tx["amount"] >= min_amount_share * threshold]

    # Cheap pre-filter: only accounts with enough distinct counterparties overall.
    counts = tx.groupby(focal_col)[other_col].nunique()
    candidates = [a for a in counts[counts >= min_counterparties].index if a not in excluded]

    alerts = []
    for acct in candidates:
        grp = tx[tx[focal_col] == acct].sort_values("timestamp")
        times = grp["timestamp"].to_numpy()
        others = grp[other_col].to_numpy()
        seen: Counter = Counter()
        best, best_span, lo = 0, (0, 0), 0
        for hi in range(len(grp)):  # sliding window maximising distinct counterparties
            seen[others[hi]] += 1
            while times[hi] - times[lo] > window:
                seen[others[lo]] -= 1
                if seen[others[lo]] == 0:
                    del seen[others[lo]]
                lo += 1
            if len(seen) > best:
                best, best_span = len(seen), (lo, hi + 1)
        if best < min_counterparties:
            continue

        win = grp.iloc[best_span[0] : best_span[1]]
        amounts = win["amount"].to_numpy()
        near = _near_threshold_share(amounts, threshold, 0.8)
        uniform = _uniformity(amounts)
        breadth = min(1.0, best / (2 * min_counterparties))
        score = 0.35 + 0.25 * breadth + 0.25 * near + 0.15 * uniform
        involved, tx_ids, cash_out = [acct, *sorted(set(win[other_col]))], list(win["tx_id"]), ""

        if direction == "in":
            # Follow the pooled money: large transfers out of the collector shortly after.
            end = win["timestamp"].max()
            outflow = tx[
                (tx["src"] == acct)
                & (tx["timestamp"] > end)
                & (tx["timestamp"] <= end + pd.Timedelta(days=followup_days))
                & (tx["amount"] >= 0.5 * amounts.sum())
            ]
            if len(outflow):
                involved += sorted(set(outflow["dst"]) - set(involved))
                tx_ids += list(outflow["tx_id"])
                cash_out = f"; {outflow['amount'].sum():,.0f} moved on to {outflow['dst'].nunique()} account(s)"

        alerts.append(
            Alert(
                typology=f"fan_{direction}",
                accounts=involved,
                tx_ids=tx_ids,
                score=round(min(score, 1.0), 3),
                details=(
                    f"{best} distinct counterparties in {window_days:g} days, "
                    f"total {amounts.sum():,.0f}, {near:.0%} of amounts just under threshold{cash_out}"
                ),
                focal=acct,
            )
        )
    return alerts


def detect_structuring(
    tx: pd.DataFrame,
    threshold: float = 10_000,
    band: float = 0.85,
    window_days: float = 14,
    min_count: int = 5,
) -> list[Alert]:
    """Repeated transfers from one account sized just below the reporting threshold."""
    near = tx[(tx["amount"] >= band * threshold) & (tx["amount"] < threshold)]
    window = pd.Timedelta(days=window_days)
    alerts = []
    for src, grp in near.groupby("src"):
        if len(grp) < min_count:
            continue
        grp = grp.sort_values("timestamp")
        times = grp["timestamp"].to_numpy()
        best, span, lo = 0, (0, 0), 0
        for hi in range(len(grp)):
            while times[hi] - times[lo] > window:
                lo += 1
            if hi - lo + 1 > best:
                best, span = hi - lo + 1, (lo, hi + 1)
        if best < min_count:
            continue
        win = grp.iloc[span[0] : span[1]]
        score = 0.5 + 0.5 * min(1.0, (best - min_count + 1) / min_count)
        alerts.append(
            Alert(
                typology="structuring",
                accounts=[src, *sorted(set(win["dst"]))],
                tx_ids=list(win["tx_id"]),
                score=round(score, 3),
                details=f"{best} transfers between {band * threshold:,.0f} and {threshold:,.0f} within {window_days:g} days",
                focal=src,
            )
        )
    return alerts


def _temporal_paths(
    tx: pd.DataFrame,
    index: TemporalIndex,
    max_hop_hours: float,
    min_ratio: float,
    max_ratio: float,
    max_len: int,
    min_amount: float,
    max_paths_per_start: int = 200,
):
    """Depth-first search for time-respecting money trails.

    A trail follows A->B at time t, then B->C after t (within ``max_hop_hours``)
    with an amount between ``min_ratio`` and ``max_ratio`` of the previous hop,
    i.e. the same money moving onward minus a small cut. Yields
    ``(kind, nodes, tx_ids, amounts)`` where kind is ``"cycle"`` or ``"chain"``.
    """
    gap = int(max_hop_hours * NS_PER_HOUR)
    starts = tx[tx["amount"] >= min_amount]
    start_times = starts["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)

    for (src, dst, amt, tid), t0 in zip(
        starts[["src", "dst", "amount", "tx_id"]].itertuples(index=False, name=None), start_times
    ):
        emitted = 0
        stack = [([src, dst], [tid], [amt], int(t0))]
        while stack and emitted < max_paths_per_start:
            nodes, tids, amts, t = stack.pop()
            extended = False
            if len(tids) < max_len:
                cur, last_amt = nodes[-1], amts[-1]
                for t_next, a_next, d_next, tid_next in index.successors(cur, t, t + gap):
                    if not (min_ratio * last_amt <= a_next <= max_ratio * last_amt):
                        continue
                    if d_next == nodes[0] and len(nodes) >= 3:
                        yield "cycle", nodes, tids + [tid_next], amts + [a_next]
                        emitted += 1
                        extended = True
                    elif d_next not in nodes:
                        stack.append((nodes + [d_next], tids + [tid_next], amts + [a_next], t_next))
                        extended = True
            if not extended:
                yield "chain", nodes, tids, amts
                emitted += 1


def detect_cycles_and_layering(
    tx: pd.DataFrame,
    index: TemporalIndex | None = None,
    max_hop_hours: float = 72,
    min_ratio: float = 0.85,
    max_ratio: float = 1.02,
    max_len: int = 8,
    min_chain_hops: int = 4,
    min_amount: float = 5_000,
) -> list[Alert]:
    """Round-tripping cycles and long layering chains, found by temporal path search.

    Static cycle enumeration on the aggregated graph explodes on real data and
    ignores time; following money forward in time with amount continuity is
    both faster and far more specific.
    """
    index = index or TemporalIndex.build(tx)
    cycles, chains = {}, []
    for kind, nodes, tids, amts in _temporal_paths(
        tx, index, max_hop_hours, min_ratio, max_ratio, max_len, min_amount
    ):
        if kind == "cycle":
            key = frozenset(tids)
            if key not in cycles:
                cycles[key] = (nodes, tids, amts)
        elif len(tids) >= min_chain_hops:
            chains.append((nodes, tids, amts))

    alerts = []
    for nodes, tids, amts in cycles.values():
        retained = amts[-1] / amts[0]
        score = 0.6 + 0.2 * min(1.0, len(nodes) / 5) + 0.2 * retained
        alerts.append(
            Alert(
                typology="cycle",
                accounts=list(nodes),
                tx_ids=list(tids),
                score=round(min(score, 1.0), 3),
                details=f"{len(nodes)}-account loop moving {amts[0]:,.0f}, {retained:.0%} returned to origin",
                focal=nodes[0],
            )
        )

    # Keep only maximal chains: drop any chain whose transactions are contained in a longer one.
    chains.sort(key=lambda c: len(c[1]), reverse=True)
    kept: list[tuple[list, list, list]] = []
    kept_sets: list[set] = []
    cycle_tx = set().union(*cycles) if cycles else set()
    for nodes, tids, amts in chains:
        s = set(tids)
        if s <= cycle_tx or any(s <= k for k in kept_sets):
            continue
        kept.append((nodes, tids, amts))
        kept_sets.append(s)

    for nodes, tids, amts in kept:
        hops = len(tids)
        score = 0.45 + 0.4 * min(1.0, (hops - min_chain_hops + 1) / 4) + 0.15 * (amts[-1] / amts[0])
        alerts.append(
            Alert(
                typology="layering",
                accounts=list(nodes),
                tx_ids=list(tids),
                score=round(min(score, 1.0), 3),
                details=f"{hops}-hop chain moving {amts[0]:,.0f} -> {amts[-1]:,.0f}",
                focal=nodes[0],
            )
        )
    return alerts


def propagate_risk(g: nx.DiGraph, seeds: dict[str, float], alpha: float = 0.85) -> pd.Series:
    """Guilt-by-association: personalised PageRank seeded with detector scores.

    Raw personalised PageRank favours hubs (popular merchants) that everyone
    touches, so the result is divided by global PageRank to get a *lift*: how
    much closer an account is to known-bad accounts than to the network at large.
    Returned values are scaled to ``[0, 1]``.
    """
    if not seeds:
        return pd.Series(0.0, index=list(g.nodes))
    ug = g.to_undirected()
    for _, _, d in ug.edges(data=True):
        d["w"] = np.log1p(d["amount"])
    ppr = pd.Series(nx.pagerank(ug, alpha=alpha, personalization=seeds, weight="w"))
    base = pd.Series(nx.pagerank(ug, alpha=alpha, weight="w"))
    lift = np.log1p(ppr / base)
    return (lift / lift.max()).fillna(0.0) if lift.max() > 0 else lift


def community_summary(g: nx.DiGraph, flagged: set[str], seed: int = 0) -> pd.DataFrame:
    """Louvain communities with their share of flagged accounts.

    Communities dense with flagged accounts point at organised networks
    rather than isolated incidents.
    """
    ug = nx.Graph()
    ug.add_nodes_from(g.nodes)
    for u, v, d in g.edges(data=True):
        w = np.log1p(d["amount"])
        ug.add_edge(u, v, weight=ug[u][v]["weight"] + w if ug.has_edge(u, v) else w)
    comms = nx.algorithms.community.louvain_communities(ug, weight="weight", seed=seed)
    rows = []
    for i, members in enumerate(comms):
        n_flag = len(members & flagged)
        rows.append({"community": i, "size": len(members), "flagged": n_flag, "flagged_share": n_flag / len(members)})
    return pd.DataFrame(rows).sort_values(["flagged", "flagged_share"], ascending=False).reset_index(drop=True)

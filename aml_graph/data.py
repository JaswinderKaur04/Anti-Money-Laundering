"""Synthetic transaction data with injected money-laundering typologies.

Real AML datasets are confidential, so this module produces a realistic-ish
background of retail, payroll and B2B payments, then plants known laundering
patterns on top of it. Every planted transaction carries a ``typology`` and
``pattern_id`` label, which lets the detectors be evaluated against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPOLOGIES = ("fan_in", "fan_out", "cycle", "layering", "structuring")


@dataclass
class GeneratorConfig:
    n_accounts: int = 1000
    days: int = 90
    start: str = "2026-01-01"
    reporting_threshold: float = 10_000.0
    patterns_per_typology: int = 6
    seed: int = 42


class _Builder:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = pd.Timestamp(cfg.start)
        self.rows: list[dict] = []

    def add(self, ts, src, dst, amount, typology="", pattern_id=-1):
        self.rows.append(
            {
                "timestamp": ts,
                "src": src,
                "dst": dst,
                "amount": round(float(amount), 2),
                "typology": typology,
                "pattern_id": pattern_id,
            }
        )

    def random_time(self, lo_day=0.0, hi_day=None):
        hi_day = self.cfg.days if hi_day is None else hi_day
        return self.start + pd.Timedelta(days=float(self.rng.uniform(lo_day, hi_day)))


def _make_accounts(b: _Builder) -> pd.DataFrame:
    n = b.cfg.n_accounts
    kinds = b.rng.choice(["individual", "business", "merchant"], size=n, p=[0.85, 0.10, 0.05])
    return pd.DataFrame({"account_id": [f"A{i:05d}" for i in range(n)], "kind": kinds})


def _normal_activity(b: _Builder, accounts: pd.DataFrame) -> None:
    rng = b.rng
    ids = accounts["account_id"].to_numpy()
    by_kind = {k: ids[accounts["kind"].to_numpy() == k] for k in ("individual", "business", "merchant")}
    individuals, businesses, merchants = by_kind["individual"], by_kind["business"], by_kind["merchant"]
    # Popular merchants receive most of the card-like spend (heavy-tailed popularity).
    merchant_p = rng.pareto(1.5, len(merchants)) + 1
    merchant_p /= merchant_p.sum()

    for acct in individuals:
        for _ in range(rng.poisson(b.cfg.days * 0.3)):  # retail purchases
            b.add(b.random_time(), acct, rng.choice(merchants, p=merchant_p), rng.lognormal(3.7, 0.9))
        for _ in range(rng.poisson(b.cfg.days * 0.04)):  # peer-to-peer transfers
            b.add(b.random_time(), acct, rng.choice(individuals), rng.lognormal(5.0, 1.0))

    # Payroll: each business pays a stable set of employees once a month.
    for biz in businesses:
        staff = rng.choice(individuals, size=rng.integers(3, 25), replace=False)
        salaries = rng.lognormal(8.0, 0.3, len(staff))
        for month_start in range(0, b.cfg.days, 30):
            day = b.start + pd.Timedelta(days=month_start + 27, hours=9)
            for emp, sal in zip(staff, salaries):
                b.add(day + pd.Timedelta(minutes=float(rng.uniform(0, 120))), biz, emp, sal)
        for _ in range(rng.poisson(b.cfg.days * 0.15)):  # B2B invoices
            b.add(b.random_time(), biz, rng.choice(businesses), rng.lognormal(9.0, 1.0))

    for m in merchants:  # merchants settle with suppliers
        for _ in range(rng.poisson(b.cfg.days * 0.1)):
            b.add(b.random_time(), m, rng.choice(businesses), rng.lognormal(8.5, 0.8))


def _inject_patterns(b: _Builder, accounts: pd.DataFrame) -> None:
    rng = b.rng
    thr = b.cfg.reporting_threshold
    individuals = accounts.loc[accounts["kind"] == "individual", "account_id"].to_numpy()
    used: set[str] = set()

    def pick(k: int) -> list[str]:
        pool = np.setdiff1d(individuals, list(used))
        chosen = list(rng.choice(pool, size=k, replace=False))
        used.update(chosen)
        return chosen

    def hop_delay():
        return pd.Timedelta(hours=float(rng.uniform(1, 40)))

    pid = 0
    for _ in range(b.cfg.patterns_per_typology):
        # Fan-in / smurfing: many mules deposit just-under-threshold sums into one collector.
        collector, exit_acct, *mules = pick(int(rng.integers(10, 20)) + 2)
        t0 = b.random_time(0, b.cfg.days - 10)
        total = 0.0
        for mule in mules:
            amt = rng.uniform(0.8, 0.99) * thr
            total += amt
            b.add(t0 + pd.Timedelta(hours=float(rng.uniform(0, 96))), mule, collector, amt, "fan_in", pid)
        b.add(t0 + pd.Timedelta(days=5), collector, exit_acct, total * 0.97, "fan_in", pid)
        pid += 1

        # Fan-out: one account rapidly disperses a large sum to many recipients.
        source, *recipients = pick(int(rng.integers(10, 20)) + 1)
        t0 = b.random_time(0, b.cfg.days - 5)
        for r in recipients:
            amt = rng.uniform(0.85, 0.99) * thr
            b.add(t0 + pd.Timedelta(hours=float(rng.uniform(0, 48))), source, r, amt, "fan_out", pid)
        pid += 1

        # Cycle / round-tripping: funds travel around a loop back to the originator.
        ring = pick(int(rng.integers(3, 7)))
        t = b.random_time(0, b.cfg.days - 15)
        amt = rng.uniform(20_000, 200_000)
        for i, src in enumerate(ring):
            b.add(t, src, ring[(i + 1) % len(ring)], amt, "cycle", pid)
            t += hop_delay()
            amt *= rng.uniform(0.95, 0.995)  # small "fee" skimmed at each hop
        pid += 1

        # Layering: a long, fast chain of pass-through accounts.
        chain = pick(int(rng.integers(5, 9)))
        t = b.random_time(0, b.cfg.days - 15)
        amt = rng.uniform(30_000, 300_000)
        for src, dst in zip(chain, chain[1:]):
            b.add(t, src, dst, amt, "layering", pid)
            t += hop_delay()
            amt *= rng.uniform(0.95, 0.995)
        pid += 1

        # Structuring: repeated transfers kept just below the reporting threshold.
        src, *dsts = pick(int(rng.integers(1, 4)) + 1)
        t0 = b.random_time(0, b.cfg.days - 15)
        for _ in range(int(rng.integers(6, 13))):
            ts = t0 + pd.Timedelta(days=float(rng.uniform(0, 14)))
            b.add(ts, src, rng.choice(dsts), rng.uniform(0.9, 0.995) * thr, "structuring", pid)
        pid += 1


def generate(cfg: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(accounts, transactions)`` DataFrames with ground-truth labels."""
    cfg = cfg or GeneratorConfig()
    b = _Builder(cfg)
    accounts = _make_accounts(b)
    _normal_activity(b, accounts)
    _inject_patterns(b, accounts)

    tx = pd.DataFrame(b.rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx.insert(0, "tx_id", [f"T{i:07d}" for i in range(len(tx))])
    bad = set(tx.loc[tx["typology"] != "", "src"]) | set(tx.loc[tx["typology"] != "", "dst"])
    accounts["is_laundering"] = accounts["account_id"].isin(bad)
    return accounts, tx

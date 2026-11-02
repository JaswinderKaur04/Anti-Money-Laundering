import pandas as pd
import pytest

from aml_graph import GeneratorConfig, generate, run
from aml_graph import detectors as det
from aml_graph.__main__ import main
from aml_graph.graph import TemporalIndex, aggregate_graph

T0 = pd.Timestamp("2026-01-01")


def make_tx(rows):
    """rows: (hours_from_t0, src, dst, amount)"""
    tx = pd.DataFrame(rows, columns=["hours", "src", "dst", "amount"])
    tx["timestamp"] = T0 + pd.to_timedelta(tx.pop("hours"), unit="h")
    tx.insert(0, "tx_id", [f"T{i}" for i in range(len(tx))])
    return tx


def make_accounts(ids, kind="individual"):
    return pd.DataFrame({"account_id": sorted(ids), "kind": kind})


def test_fan_in_detects_smurfing_and_follows_cash_out():
    rows = [(i * 5, f"M{i}", "C", 9_500) for i in range(10)]
    rows.append((24 * 8, "C", "EXIT", 90_000))
    tx = make_tx(rows)
    alerts = det.detect_fan(tx, make_accounts(set(tx.src) | set(tx.dst)), "in")
    assert len(alerts) == 1
    a = alerts[0]
    assert a.focal == "C" and "EXIT" in a.accounts
    assert len(a.tx_ids) == 11
    assert a.score > 0.8


def test_fan_in_ignores_slow_or_small_inflows():
    spread_out = make_tx([(i * 24 * 3, f"M{i}", "C", 9_500) for i in range(10)])  # one every 3 days
    small = make_tx([(i, f"M{i}", "C", 50) for i in range(20)])  # card-sized payments
    for tx in (spread_out, small):
        assert det.detect_fan(tx, make_accounts(set(tx.src) | set(tx.dst)), "in") == []


def test_fan_in_skips_merchants():
    tx = make_tx([(i, f"M{i}", "SHOP", 3_000) for i in range(15)])
    accounts = pd.concat([make_accounts(set(tx.src)), make_accounts({"SHOP"}, "merchant")])
    assert det.detect_fan(tx, accounts, "in") == []


def test_fan_out():
    tx = make_tx([(i, "S", f"R{i}", 9_000) for i in range(12)])
    alerts = det.detect_fan(tx, make_accounts(set(tx.src) | set(tx.dst)), "out")
    assert [a.focal for a in alerts] == ["S"]


def test_structuring():
    tx = make_tx([(i * 30, "S", "D", 9_800) for i in range(6)] + [(5, "X", "D", 9_800)])
    alerts = det.detect_structuring(tx)
    assert [a.focal for a in alerts] == ["S"]
    assert len(alerts[0].tx_ids) == 6


def test_cycle_requires_time_order_and_amount_continuity():
    good = make_tx([(0, "A", "B", 50_000), (5, "B", "C", 49_000), (10, "C", "D", 48_000), (15, "D", "A", 47_500)])
    alerts = det.detect_cycles_and_layering(good)
    cycles = [a for a in alerts if a.typology == "cycle"]
    assert len(cycles) == 1 and set(cycles[0].accounts) == {"A", "B", "C", "D"}
    assert not [a for a in alerts if a.typology == "layering"]  # chain is subsumed by the cycle

    # Same edges, but the closing transfer happens *before* the money arrives: no cycle.
    backwards = make_tx([(10, "A", "B", 50_000), (15, "B", "C", 49_000), (0, "C", "A", 48_000)])
    assert not [a for a in det.detect_cycles_and_layering(backwards) if a.typology == "cycle"]

    # Amount collapses mid-loop: not the same money.
    leaky = make_tx([(0, "A", "B", 50_000), (5, "B", "C", 10_000), (10, "C", "A", 9_900)])
    assert not [a for a in det.detect_cycles_and_layering(leaky) if a.typology == "cycle"]


def test_layering_reports_only_maximal_chain():
    nodes = list("ABCDEFG")
    tx = make_tx([(i * 6, s, d, 100_000 * 0.98**i) for i, (s, d) in enumerate(zip(nodes, nodes[1:]))])
    alerts = [a for a in det.detect_cycles_and_layering(tx) if a.typology == "layering"]
    assert len(alerts) == 1
    assert alerts[0].accounts == nodes


def test_temporal_index_window():
    tx = make_tx([(1, "A", "B", 1), (5, "A", "C", 1), (50, "A", "D", 1)])
    idx = TemporalIndex.build(tx)
    t1 = int(tx.timestamp[0].value)
    got = [d for _, _, d, _ in idx.successors("A", t1, t1 + 10 * 3_600_000_000_000)]
    assert got == ["C"]


def test_propagation_favours_neighbours_of_seed():
    tx = make_tx([(0, "BAD", "N1", 5_000), (1, "N1", "N2", 5_000), (2, "X", "Y", 5_000), (3, "Y", "Z", 5_000)])
    risk = det.propagate_risk(aggregate_graph(tx), {"BAD": 1.0})
    assert risk["N1"] > risk["N2"] > risk["Y"]


@pytest.fixture(scope="module")
def small_run():
    acc, tx = generate(GeneratorConfig(n_accounts=400, days=60, patterns_per_typology=3, seed=7))
    return acc, tx, run(tx, acc)


def test_generator_labels(small_run):
    acc, tx, _ = small_run
    assert set(tx.loc[tx.typology != "", "typology"]) == set(("fan_in", "fan_out", "cycle", "layering", "structuring"))
    assert acc.is_laundering.sum() > 0
    assert tx.tx_id.is_unique


def test_pipeline_catches_planted_patterns(small_run):
    _, _, result = small_run
    m = result.metrics
    for typ, stats in m["pattern_recall"].items():
        assert stats["recall"] == 1.0, typ
    assert m["precision"] > 0.9 and m["recall"] > 0.9


def test_cli_roundtrip(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    main(["generate", "--accounts", "300", "--days", "45", "--patterns", "2", "--out", str(data)])
    main(["detect", "--data", str(data), "--out", str(out)])
    for name in ("alerts.csv", "account_scores.csv", "report.md", "network.html", "metrics.json",
                 "suspicious_subgraph.graphml", "communities.csv"):
        assert (out / name).exists(), name

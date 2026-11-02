"""Command line interface.

    python -m aml_graph generate --out data/
    python -m aml_graph detect --data data/ --out output/
    python -m aml_graph run --out output/          # generate + detect in one go
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .data import GeneratorConfig, generate
from .pipeline import PipelineConfig, run
from .report import write_report


def _generate(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = GeneratorConfig(
        n_accounts=args.accounts, days=args.days, patterns_per_typology=args.patterns, seed=args.seed
    )
    accounts, tx = generate(cfg)
    print(f"Generated {len(accounts):,} accounts and {len(tx):,} transactions "
          f"({(tx['typology'] != '').sum():,} planted laundering transactions)")
    return accounts, tx


def _detect(accounts, tx, out_dir: str, pcfg: PipelineConfig) -> None:
    t0 = time.perf_counter()
    result = run(tx, accounts, pcfg)
    out = write_report(result, out_dir)
    print(f"Pipeline finished in {time.perf_counter() - t0:.1f}s: "
          f"{len(result.alerts)} alerts, {int(result.scores['flagged'].sum())} accounts flagged")
    if result.metrics:
        m = result.metrics
        print(f"Precision {m['precision']:.1%}  Recall {m['recall']:.1%}  F1 {m['f1']:.3f}  "
              f"Precision@k {m['precision_at_k']:.1%}")
        for typ, st in sorted(m["pattern_recall"].items()):
            print(f"  {typ:<12} caught {st['caught']}/{st['patterns']}")
    print(f"Report written to {out.resolve()} (open network.html for the interactive graph)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="aml_graph", description="Anti-money-laundering detection with graph analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    def gen_args(sp):
        sp.add_argument("--accounts", type=int, default=1000)
        sp.add_argument("--days", type=int, default=90)
        sp.add_argument("--patterns", type=int, default=6, help="planted patterns per typology")
        sp.add_argument("--seed", type=int, default=42)

    def det_args(sp):
        sp.add_argument("--threshold", type=float, default=10_000, help="cash reporting threshold")
        sp.add_argument("--alert-score", type=float, default=0.5, help="risk score needed to flag an account")

    g = sub.add_parser("generate", help="create a synthetic labelled dataset")
    gen_args(g)
    g.add_argument("--out", default="data")

    d = sub.add_parser("detect", help="run detection on accounts.csv + transactions.csv")
    d.add_argument("--data", default="data", help="folder with accounts.csv and transactions.csv")
    d.add_argument("--out", default="output")
    det_args(d)

    r = sub.add_parser("run", help="generate a dataset and run detection on it")
    gen_args(r)
    det_args(r)
    r.add_argument("--out", default="output")

    args = p.parse_args(argv)
    if args.cmd == "generate":
        accounts, tx = _generate(args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        accounts.to_csv(out / "accounts.csv", index=False)
        tx.to_csv(out / "transactions.csv", index=False)
        print(f"Saved to {out.resolve()}")
        return

    pcfg = PipelineConfig(threshold=args.threshold, alert_score=args.alert_score)
    if args.cmd == "detect":
        data = Path(args.data)
        accounts = pd.read_csv(data / "accounts.csv")
        tx = pd.read_csv(data / "transactions.csv", parse_dates=["timestamp"], keep_default_na=False)
        _detect(accounts, tx, args.out, pcfg)
    else:
        accounts, tx = _generate(args)
        _detect(accounts, tx, args.out, pcfg)


if __name__ == "__main__":
    main()

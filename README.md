# Anti-Money Laundering (AML) with Graph Analysis

Detects money-laundering patterns in transaction data by modelling accounts as
nodes and transfers as edges, then searching the graph for the shapes that
laundering leaves behind.

Real AML data is confidential, so the project ships with a synthetic generator
that produces ordinary retail, payroll and B2B traffic and plants labelled
laundering patterns on top of it. That ground truth lets every detector be
measured.

## Quick start

```bash
pip install -r requirements.txt
python -m aml_graph run --out output
```

Then open `output/network.html` in a browser for an interactive view of the
suspicious network, or read `output/report.md`.

Other commands:

```bash
python -m aml_graph generate --accounts 5000 --patterns 20 --out data   # write data/accounts.csv + transactions.csv
python -m aml_graph detect --data data --out output                     # analyse a folder of CSVs
python -m pytest
```

`detect` works on your own data if it has the same columns:

| File | Required columns |
|---|---|
| `accounts.csv` | `account_id`, `kind` (`individual` / `business` / `merchant`) |
| `transactions.csv` | `tx_id`, `timestamp`, `src`, `dst`, `amount` |

Add `is_laundering` (accounts) and `typology` / `pattern_id` (transactions) to
get evaluation metrics.

## Typologies detected

| Typology | What it looks like in the graph | Detector |
|---|---|---|
| **Fan-in / smurfing** | Many accounts send just-under-threshold sums to one collector within days, which then moves the pool on | Sliding window of distinct senders per account, scored by breadth, near-threshold share and amount uniformity; follows the collector's cash-out |
| **Fan-out** | One account rapidly disperses a sum to many recipients | Same window, outgoing direction |
| **Cycle / round-tripping** | Money leaves an account and comes back through a loop | Time-respecting path search: each hop must happen after the previous one, within 72h, keeping 85–102% of the amount |
| **Layering** | Long, fast chains of pass-through accounts | Same temporal search; only maximal chains of 4+ hops are reported |
| **Structuring** | Repeated transfers sized just below the reporting threshold | Count of 85–100%-of-threshold transfers per sender in a 14-day window |

Why temporal paths instead of `nx.simple_cycles`? On the aggregated graph,
ordinary traffic creates an enormous number of cycles that ignore time. Following
the *same money* forward in time (ordered timestamps, amount continuity) is both
much faster and far more specific.

## Scoring

1. Each alert has a score in `[0, 1]`.
2. **Detector score** per account combines all its alerts with a noisy-OR.
3. **Association score**: personalised PageRank seeded with detector scores,
   divided by global PageRank so hubs such as popular merchants are not
   flagged just for being well connected.
4. **Risk score** = noisy-OR of detector score and `0.4 × association`.
   Association alone can't flag an account (it caps at 0.4); it ranks
   unflagged accounts as leads for investigation.
5. Accounts with risk ≥ 0.5 are flagged. Louvain communities are ranked by how
   many flagged members they contain to highlight organised networks.

## Outputs

| File | Contents |
|---|---|
| `alerts.csv` | One row per alert: typology, score, accounts, transaction ids, explanation |
| `account_scores.csv` | Every account with flow features, PageRank, pass-through ratio and scores |
| `communities.csv` | Louvain communities and their flagged share |
| `report.md` | Summary, evaluation and top-risk accounts |
| `metrics.json` | Precision, recall, F1, precision@k, per-typology pattern recall |
| `network.html` | Interactive network of flagged accounts (needs internet for the vis-network library) |
| `suspicious_subgraph.graphml` | Same subgraph for Gephi / Cytoscape |

## Results on synthetic data

Default run (1,000 accounts, ~32k transactions, 30 planted patterns): every
planted pattern is caught, with account-level precision and recall of about
99–100%. The same holds at 5,000 accounts / 160k transactions (about 8 seconds).

These numbers are high because the planted patterns are clean and the
background traffic is simple. Real data is far noisier: expect to tune the
thresholds in `PipelineConfig` and the detector arguments, and to replace the
`exclude_kinds` shortcut with per-segment baselines.

## Project layout

```
aml_graph/
  data.py       synthetic generator with labelled typologies
  graph.py      aggregated graph, node features, temporal index
  detectors.py  fan-in/out, structuring, cycles & layering, risk propagation, communities
  pipeline.py   detectors -> account scores -> evaluation
  report.py     CSV / Markdown / GraphML / HTML output
  __main__.py   command line interface
tests/          unit tests for each detector and an end-to-end run
```

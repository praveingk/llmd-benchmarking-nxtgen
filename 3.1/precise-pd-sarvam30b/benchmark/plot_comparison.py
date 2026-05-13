#!/usr/bin/env python3
"""Plot PD (1P:1D, TP=2 each) vs baseline (2 monolithic replicas TP=2) on
sarvamai/sarvam-30b (MoE, 2.4B active / 30B total) across 4 H100 NVL GPUs.
Workload: shared_prefix synthetic, ISL=15k (10k shared + 5k unique), OSL=600.
"""
import json
import pathlib
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent
PD = next(BASE.glob("results-llmd-pd/*/inference-perf_*_llmd-pd-sarvam"))
BL = next(BASE.glob("results-baseline/*/inference-perf_*_baseline-sarvam"))

RATES = [1, 2, 3, 4, 5, 6, 8]


def load_stage(root: pathlib.Path, i: int) -> dict:
    with open(root / f"stage_{i}_lifecycle_metrics.json") as f:
        return json.load(f)


rows = []
for i, r in enumerate(RATES):
    p = load_stage(PD, i)["successes"]
    b = load_stage(BL, i)["successes"]
    rows.append({
        "rate": r,
        "pd_ttft": p["latency"]["time_to_first_token"]["mean"],
        "bl_ttft": b["latency"]["time_to_first_token"]["mean"],
        "pd_tok": p["throughput"]["output_tokens_per_sec"],
        "bl_tok": b["throughput"]["output_tokens_per_sec"],
        "pd_itl": p["latency"].get("inter_token_latency", {}).get("mean", 0) * 1000,
        "bl_itl": b["latency"].get("inter_token_latency", {}).get("mean", 0) * 1000,
    })

rates = [r["rate"] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(rates, [r["pd_ttft"] for r in rows], "o-", label="PD (1P:1D, TP=2)", linewidth=2, markersize=8)
ax.plot(rates, [r["bl_ttft"] for r in rows], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS")
ax.set_ylabel("Mean TTFT (seconds)")
ax.set_title("Time To First Token\nsarvam-30b, ISL=15k, OSL=600")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

ax = axes[1]
ax.plot(rates, [r["pd_tok"] for r in rows], "o-", label="PD (1P:1D, TP=2)", linewidth=2, markersize=8)
ax.plot(rates, [r["bl_tok"] for r in rows], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS")
ax.set_ylabel("Output tokens/sec")
ax.set_title("Output Throughput\nsarvam-30b, 4× H100 NVL")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(rates, [r["pd_itl"] for r in rows], "o-", label="PD (1P:1D, TP=2)", linewidth=2, markersize=8)
ax.plot(rates, [r["bl_itl"] for r in rows], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS")
ax.set_ylabel("Inter-token latency (ms)")
ax.set_title("Inter-Token Latency\n(PD wins: decode uncontended)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = BASE / "comparison_pd_vs_baseline.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'rate':<5} {'pd_TTFT':>8} {'bl_TTFT':>8} {'TTFT_x':>7}   {'pd_tok':>8} {'bl_tok':>8} {'tok_%':>7}   {'pd_ITL':>7} {'bl_ITL':>7}")
print("-" * 100)
for r in rows:
    ttft_x = r["pd_ttft"] / r["bl_ttft"] if r["bl_ttft"] else 0
    tok_pct = (r["pd_tok"] - r["bl_tok"]) / r["bl_tok"] * 100 if r["bl_tok"] else 0
    print(f"{r['rate']:<5} {r['pd_ttft']:>7.2f}s {r['bl_ttft']:>7.2f}s {ttft_x:>6.1f}x   "
          f"{r['pd_tok']:>8.0f} {r['bl_tok']:>8.0f} {tok_pct:>+6.1f}%   "
          f"{r['pd_itl']:>6.1f}ms {r['bl_itl']:>6.1f}ms")

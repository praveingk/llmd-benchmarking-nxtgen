#!/usr/bin/env python3
"""Balanced workload PD vs baseline, prefix-caching ON on PD prefill.
ISL=3k (1500 shared + 1500 unique), OSL=2000. sarvamai/sarvam-30b, 4× H100 NVL.

This is the third experiment in this benchmark suite:
  benchmark/                 ISL=15k, OSL=600        — prefill-dominated, PD saturated instantly
  benchmark/decode-heavy/    ISL=2k,  OSL=2k         — decode-dominated, no prefix cache
  benchmark/balanced-pc/     ISL=3k,  OSL=2k, PC=ON  — ratio-matched, prefix cache on prefill
"""
import json
import pathlib
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent
PD = next(BASE.glob("results-llmd-pd/**/inference-perf_*_llmd-pd-sarvam"))
BL = next(BASE.glob("results-baseline/**/inference-perf_*_baseline-sarvam"))
RATES = [1, 2, 3, 5, 7, 10, 12, 15]


def load(root):
    rows = []
    for i, r in enumerate(RATES):
        with open(root / f"stage_{i}_lifecycle_metrics.json") as f:
            D = json.load(f)
        s = D["successes"]
        rows.append({
            "rate": r,
            "ttft": s["latency"]["time_to_first_token"]["mean"],
            "tok": s["throughput"]["output_tokens_per_sec"],
            "itl": s["latency"].get("inter_token_latency", {}).get("mean", 0) * 1000,
        })
    return rows


pd, bl = load(PD), load(BL)
rates = [r["rate"] for r in pd]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(rates, [r["ttft"] for r in pd], "o-", label="PD (1P:1D, TP=2, prefix-cache ON)", linewidth=2, markersize=8)
axes[0].plot(rates, [r["ttft"] for r in bl], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
axes[0].set_xlabel("Target QPS"); axes[0].set_ylabel("Mean TTFT (seconds)")
axes[0].set_title("Time To First Token\nsarvam-30b, ISL=3k, OSL=2k")
axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(rates, [r["tok"] for r in pd], "o-", label="PD (1P:1D, TP=2, prefix-cache ON)", linewidth=2, markersize=8)
axes[1].plot(rates, [r["tok"] for r in bl], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
axes[1].set_xlabel("Target QPS"); axes[1].set_ylabel("Output tokens/sec")
axes[1].set_title("Output Throughput\n4× H100 NVL, shared_prefix workload")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(rates, [r["itl"] for r in pd], "o-", label="PD (1P:1D, TP=2, prefix-cache ON)", linewidth=2, markersize=8)
axes[2].plot(rates, [r["itl"] for r in bl], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
axes[2].set_xlabel("Target QPS"); axes[2].set_ylabel("Inter-token latency (ms)")
axes[2].set_title("Inter-Token Latency")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = BASE / "comparison_balanced_pc.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'rate':<5} {'pd_TTFT':>8} {'bl_TTFT':>8} {'TTFT_x':>7}   {'pd_tok':>7} {'bl_tok':>7} {'tok%':>6}   {'pd_ITL':>7} {'bl_ITL':>7}")
print("-" * 100)
for p, b in zip(pd, bl):
    tx = p["ttft"] / b["ttft"] if b["ttft"] else 0
    pct = (p["tok"] - b["tok"]) / b["tok"] * 100 if b["tok"] else 0
    print(f"{p['rate']:<5} {p['ttft']:>7.2f}s {b['ttft']:>7.2f}s {tx:>6.1f}x   "
          f"{p['tok']:>7.0f} {b['tok']:>7.0f} {pct:>+5.1f}%   "
          f"{p['itl']:>6.1f}ms {b['itl']:>6.1f}ms")

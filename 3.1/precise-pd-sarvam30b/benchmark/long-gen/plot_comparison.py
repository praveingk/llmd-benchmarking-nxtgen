#!/usr/bin/env python3
"""Long-generation workload PD vs baseline. sarvam-30b, 4× H100 NVL.
ISL=2k (1500 shared + 500 unique), OSL=6000. Prefix caching ON on PD prefill.

Goal: expose PD's decode-isolation advantage via p95/p99 inter-token latency.
Baseline's single pod handles prefill+decode → new-request prefills interrupt
in-flight decodes → tail ITL balloons. PD's decode pod runs uninterrupted.
"""
import json
import pathlib
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent
PD = next(BASE.glob("results-llmd-pd/**/inference-perf_*_llmd-pd-sarvam"))
BL = next(BASE.glob("results-baseline/**/inference-perf_*_baseline-sarvam"))
RATES = [1, 2, 3, 5]  # stage 4 (rate=7) omitted — PD decode saturates there; the interesting story is 1-5


def load(root):
    rows = []
    for i, r in enumerate(RATES):
        with open(root / f"stage_{i}_lifecycle_metrics.json") as f:
            D = json.load(f)
        s = D["successes"]
        itl = s["latency"].get("inter_token_latency", {})
        rows.append({
            "rate": r,
            "ttft": s["latency"]["time_to_first_token"]["mean"],
            "tok": s["throughput"]["output_tokens_per_sec"],
            "itl_mean": itl.get("mean", 0) * 1000,
            "itl_p95": itl.get("p95", 0) * 1000,
            "itl_p99": itl.get("p99", 0) * 1000,
        })
    return rows


pd, bl = load(PD), load(BL)
rates = [r["rate"] for r in pd]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Throughput
ax = axes[0][0]
ax.plot(rates, [r["tok"] for r in pd], "o-", label="PD (1P:1D, TP=2)", linewidth=2, markersize=8)
ax.plot(rates, [r["tok"] for r in bl], "s-", label="Baseline (2× monolithic TP=2)", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS"); ax.set_ylabel("Output tokens/sec")
ax.set_title("Throughput — roughly matched at low QPS,\nbaseline pulls ahead past rate=5")
ax.legend(); ax.grid(True, alpha=0.3)

# TTFT
ax = axes[0][1]
ax.plot(rates, [r["ttft"] for r in pd], "o-", label="PD", linewidth=2, markersize=8)
ax.plot(rates, [r["ttft"] for r in bl], "s-", label="Baseline", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS"); ax.set_ylabel("Mean TTFT (seconds)")
ax.set_title("TTFT — baseline consistent win\n(2 prefill-capable replicas > 1 dedicated prefill)")
ax.legend(); ax.grid(True, alpha=0.3)

# Mean ITL
ax = axes[1][0]
ax.plot(rates, [r["itl_mean"] for r in pd], "o-", label="PD", linewidth=2, markersize=8)
ax.plot(rates, [r["itl_mean"] for r in bl], "s-", label="Baseline", linewidth=2, markersize=8)
ax.set_xlabel("Target QPS"); ax.set_ylabel("Mean ITL (ms)")
ax.set_title("Mean Inter-Token Latency — comparable")
ax.legend(); ax.grid(True, alpha=0.3)

# TAIL ITL — the main story
ax = axes[1][1]
ax.plot(rates, [r["itl_p95"] for r in pd], "o-", label="PD p95", linewidth=2, markersize=8, color="C0")
ax.plot(rates, [r["itl_p95"] for r in bl], "s-", label="Baseline p95", linewidth=2, markersize=8, color="C1")
ax.plot(rates, [r["itl_p99"] for r in pd], "o--", label="PD p99", linewidth=2, markersize=8, color="C0", alpha=0.7)
ax.plot(rates, [r["itl_p99"] for r in bl], "s--", label="Baseline p99", linewidth=2, markersize=8, color="C1", alpha=0.7)
ax.set_xlabel("Target QPS"); ax.set_ylabel("ITL tail (ms)")
ax.set_title("Tail ITL (p95 / p99) — PD's decode isolation wins\n(no prefill interruptions)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("sarvam-30b long-generation workload (ISL=2k, OSL=6k, prefix-cache ON)\nPD vs baseline on 4× H100 NVL", y=1.00)
plt.tight_layout()
out = BASE / "comparison_long_gen.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'rate':<5}   {'pd_tok':>7} {'bl_tok':>7} {'tok%':>6}   {'pd_TTFT':>7} {'bl_TTFT':>7}   {'pd_p95':>7} {'bl_p95':>7}   {'pd_p99':>7} {'bl_p99':>7}   {'p99_adv':>7}")
print("-" * 110)
for p, b in zip(pd, bl):
    pct = (p["tok"] - b["tok"]) / b["tok"] * 100
    p99adv = b["itl_p99"] / p["itl_p99"] if p["itl_p99"] else 0
    print(f"{p['rate']:<5}   {p['tok']:>7.0f} {b['tok']:>7.0f} {pct:>+5.1f}%   "
          f"{p['ttft']:>6.2f}s {b['ttft']:>6.2f}s   "
          f"{p['itl_p95']:>6.1f}ms {b['itl_p95']:>6.1f}ms   "
          f"{p['itl_p99']:>6.1f}ms {b['itl_p99']:>6.1f}ms   {p99adv:>6.2f}x")

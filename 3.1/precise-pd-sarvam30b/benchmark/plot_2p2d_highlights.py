#!/usr/bin/env python3
"""Focused PD vs baseline plot: throughput + p99 ITL for balanced and prefill-heavy.
Skips long-gen (baseline wins all metrics there). Shows the two workloads where PD
has a real case (p99 ITL advantage on prefill-contended workloads).
"""
import json
import pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent

RUNS = [
    ("2p2d-balanced",       [1, 2, 3, 5, 7, 10, 12],     "Balanced (ISL=3k, OSL=2k)"),
    ("2p2d-prefill-heavy",  [1, 2, 3, 5, 7],             "Prefill-heavy (ISL=8k, OSL=500)"),
]


def load(sub: str, tag: str, rates):
    files = sorted(
        (ROOT / sub).glob(f"results-{tag}/**/stage_*_lifecycle_metrics.json"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    out = []
    for f in files:
        with open(f) as j:
            D = json.load(j)
        i = int(f.name.split("_")[1])
        if i >= len(rates):
            continue
        s = D["successes"]
        itl = s["latency"].get("inter_token_latency", {})
        out.append({
            "rate": rates[i],
            "tok": s["throughput"]["output_tokens_per_sec"],
            "p99": itl.get("p99", 0) * 1000,
        })
    return out


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row, (sub, rates, title) in enumerate(RUNS):
    pd = load(sub, "llmd-pd", rates)
    bl = load(sub, "baseline", rates)
    xs = [r["rate"] for r in pd]

    ax = axes[row][0]
    ax.plot(xs, [r["tok"] for r in pd], "o-", label="llm-d PD (2P:2D TP=1)", linewidth=2, markersize=8)
    ax.plot(xs, [r["tok"] for r in bl], "s-", label="plain k8s (4×TP=1)", linewidth=2, markersize=8)
    ax.set_xlabel("Target QPS")
    ax.set_ylabel("Output tokens/sec")
    ax.set_title(f"{title}\nThroughput")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[row][1]
    ax.plot(xs, [r["p99"] for r in pd], "o-", label="llm-d PD p99", linewidth=2, markersize=8)
    ax.plot(xs, [r["p99"] for r in bl], "s-", label="plain k8s p99", linewidth=2, markersize=8)
    ax.set_xlabel("Target QPS")
    ax.set_ylabel("p99 ITL (ms)")
    ax.set_title(f"{title}\np99 Inter-Token Latency")
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("sarvam-30b: llm-d PD (2P:2D TP=1) vs plain-k8s (4×TP=1) — 4× H100 NVL", y=1.00, fontsize=13)
plt.tight_layout()
out = ROOT / "comparison_2p2d_highlights.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

#!/usr/bin/env python3
"""Generate 3 comparison plots for the 2P:2D PD vs baseline 4xTP=1 suite.
Each subdir has results-llmd-pd/ and results-baseline/.
"""
import json
import pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent

RUNS = [
    ("2p2d-long-gen",       [1, 2, 3, 5, 7, 10, 12],   "Long-gen (ISL=2k, OSL=6k)"),
    ("2p2d-balanced",       [1, 2, 3, 5, 7, 10, 12, 15], "Balanced (ISL=3k, OSL=2k)"),
    ("2p2d-prefill-heavy",  [1, 2, 3, 5, 7],           "Prefill-heavy (ISL=8k, OSL=500)"),
]


def load(sub: str, tag: str, rates: list[int]):
    files = sorted(
        (ROOT / sub).glob(f"results-{tag}/**/stage_*_lifecycle_metrics.json"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    rows = []
    for f in files:
        with open(f) as j:
            D = json.load(j)
        i = int(f.name.split("_")[1])
        if i >= len(rates):
            continue
        s = D["successes"]
        itl = s["latency"].get("inter_token_latency", {})
        rows.append({
            "rate": rates[i],
            "ttft": s["latency"]["time_to_first_token"]["mean"],
            "tok": s["throughput"]["output_tokens_per_sec"],
            "p95": itl.get("p95", 0) * 1000,
            "p99": itl.get("p99", 0) * 1000,
        })
    return rows


for sub, rates, title in RUNS:
    pd = load(sub, "llmd-pd", rates)
    bl = load(sub, "baseline", rates)
    if not pd or not bl:
        print(f"skip {sub}: missing data")
        continue
    xs = [r["rate"] for r in pd]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"sarvam-30b 2P:2D PD vs 4×TP=1 baseline — {title}", y=1.00)

    ax = axes[0]
    ax.plot(xs, [r["tok"] for r in pd], "o-", label="PD (2P:2D TP=1, NIXL+EPP)", linewidth=2, markersize=8)
    ax.plot(xs, [r["tok"] for r in bl], "s-", label="Baseline (4 monolithic TP=1, plain k8s)", linewidth=2, markersize=8)
    ax.set_xlabel("Target QPS"); ax.set_ylabel("Output tokens/sec")
    ax.set_title("Throughput"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, [r["ttft"] for r in pd], "o-", label="PD", linewidth=2, markersize=8)
    ax.plot(xs, [r["ttft"] for r in bl], "s-", label="Baseline", linewidth=2, markersize=8)
    ax.set_xlabel("Target QPS"); ax.set_ylabel("Mean TTFT (s)")
    ax.set_title("Time To First Token")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(xs, [r["p95"] for r in pd], "o-", label="PD p95", linewidth=2, markersize=8, color="C0")
    ax.plot(xs, [r["p95"] for r in bl], "s-", label="Baseline p95", linewidth=2, markersize=8, color="C1")
    ax.plot(xs, [r["p99"] for r in pd], "o--", label="PD p99", linewidth=2, markersize=8, color="C0", alpha=0.7)
    ax.plot(xs, [r["p99"] for r in bl], "s--", label="Baseline p99", linewidth=2, markersize=8, color="C1", alpha=0.7)
    ax.set_xlabel("Target QPS"); ax.set_ylabel("ITL tail (ms)")
    ax.set_title("Inter-Token Latency (p95 / p99)"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ROOT / sub / "comparison_2p2d.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close()

    print(f"\n=== {title} ===")
    print(f"{'rate':<4} {'pd_tok':>7} {'bl_tok':>7} {'tok%':>6}   {'pd_TTFT':>7} {'bl_TTFT':>7}   {'pd_p99':>7} {'bl_p99':>7} {'p99_x':>6}")
    print("-" * 90)
    for p, b in zip(pd, bl):
        pct = (p["tok"] - b["tok"]) / b["tok"] * 100
        px = b["p99"] / p["p99"] if p["p99"] else 0
        print(f"{p['rate']:<4} {p['tok']:>7.0f} {b['tok']:>7.0f} {pct:>+5.1f}%   "
              f"{p['ttft']:>6.2f}s {b['ttft']:>6.2f}s   "
              f"{p['p99']:>6.1f}ms {b['p99']:>6.1f}ms {px:>5.2f}x")
    print()

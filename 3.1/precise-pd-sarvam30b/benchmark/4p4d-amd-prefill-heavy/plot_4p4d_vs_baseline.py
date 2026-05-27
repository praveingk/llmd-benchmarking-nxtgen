#!/usr/bin/env python3
"""4P×TP=1 + 4D×TP=1 vs 8×TP=1 round-robin baseline on a single MI325X 8-GPU node.
Workload: prefill-heavy (sys=6000, q=2000, osl=500), rate ladder 1/5/10/25/50/75/100/125/150.
Plots: output throughput, TTFT p50, E2E request latency p50, ITL p50.
"""
import json
import pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
RATES = [1, 5, 10, 25, 50, 75, 100, 125, 150]


def load(results_dir: pathlib.Path):
    files = sorted(
        results_dir.glob("**/stage_*_lifecycle_metrics.json"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    out = []
    for f in files:
        with open(f) as j:
            D = json.load(j)
        i = int(f.name.split("_")[1])
        if i >= len(RATES):
            continue
        s = D["successes"]["latency"]
        tput = D["successes"]["throughput"]
        out.append({
            "rate": RATES[i],
            "tok_s": tput["output_tokens_per_sec"],
            "ttft_p50": s["time_to_first_token"]["median"],
            "e2e_p50":  s["request_latency"]["median"],
            "itl_p50":  s["inter_token_latency"]["median"] * 1000,  # → ms
        })
    return out


pd = load(ROOT / "results-llmd-pd")
bl = load(ROOT / "results-baseline")

xs = [r["rate"] for r in pd]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

PD_LABEL = "llm-d PD (4P×TP1 + 4D×TP1)"
BL_LABEL = "plain k8s (8×TP1 round-robin)"

# Top-left: throughput
ax = axes[0][0]
ax.plot(xs, [r["tok_s"] for r in pd], "o-", label=PD_LABEL, linewidth=2, markersize=8)
ax.plot(xs, [r["tok_s"] for r in bl], "s-", label=BL_LABEL, linewidth=2, markersize=8)
ax.set_xlabel("Target request rate (rps)")
ax.set_ylabel("Output throughput (tokens/sec)")
ax.set_title("Output throughput")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Top-right: TTFT p50
ax = axes[0][1]
ax.plot(xs, [r["ttft_p50"] for r in pd], "o-", label=PD_LABEL, linewidth=2, markersize=8)
ax.plot(xs, [r["ttft_p50"] for r in bl], "s-", label=BL_LABEL, linewidth=2, markersize=8)
ax.set_xlabel("Target request rate (rps)")
ax.set_ylabel("TTFT p50 (s)")
ax.set_title("Time-to-First-Token (p50)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Bottom-left: E2E request latency p50
ax = axes[1][0]
ax.plot(xs, [r["e2e_p50"] for r in pd], "o-", label=PD_LABEL, linewidth=2, markersize=8)
ax.plot(xs, [r["e2e_p50"] for r in bl], "s-", label=BL_LABEL, linewidth=2, markersize=8)
ax.set_xlabel("Target request rate (rps)")
ax.set_ylabel("Request latency p50 (s)")
ax.set_title("End-to-end Request Latency (p50)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Bottom-right: ITL p50
ax = axes[1][1]
ax.plot(xs, [r["itl_p50"] for r in pd], "o-", label=PD_LABEL, linewidth=2, markersize=8)
ax.plot(xs, [r["itl_p50"] for r in bl], "s-", label=BL_LABEL, linewidth=2, markersize=8)
ax.set_xlabel("Target request rate (rps)")
ax.set_ylabel("ITL p50 (ms)")
ax.set_title("Inter-Token Latency (p50)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.suptitle(
    "sarvam-30b on 1× MI325X (8 GPUs): 4P×TP1 + 4D×TP1 vs 8×TP1 baseline",
    y=1.00, fontsize=13,
)
plt.tight_layout()
out = ROOT / "comparison_4p4d_vs_baseline.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

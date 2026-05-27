#!/usr/bin/env python3
"""4P×TP=1 + 4D×TP=1 vs 8×TP=1 round-robin baseline on a single MI325X 8-GPU node.
Workload: prefill-heavy (sys=6000, q=2000, osl=500), rate ladder 1/5/10/25/50/75/100/125/150.
Plots: output throughput, plus TTFT / E2E / ITL with p50 (solid) and p95 (dashed) overlaid.
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
            "ttft_p95": s["time_to_first_token"]["p95"],
            "e2e_p50":  s["request_latency"]["median"],
            "e2e_p95":  s["request_latency"]["p95"],
            "itl_p50":  s["inter_token_latency"]["median"] * 1000,
            "itl_p95":  s["inter_token_latency"]["p95"]    * 1000,
        })
    return out


pd = load(ROOT / "results-llmd-pd")
bl = load(ROOT / "results-baseline")

xs = [r["rate"] for r in pd]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

PD_NAME = "llm-d PD (4P×TP1 + 4D×TP1)"
BL_NAME = "plain k8s (8×TP1 round-robin)"
PD_C, BL_C = "C0", "C1"


def overlay(ax, key50, key95, ylabel, title, legend_loc="upper left"):
    ax.plot(xs, [r[key50] for r in pd], "o-",  color=PD_C, linewidth=2, markersize=8,
            label=f"{PD_NAME} p50")
    ax.plot(xs, [r[key95] for r in pd], "o--", color=PD_C, linewidth=2, markersize=8,
            alpha=0.7, label=f"{PD_NAME} p95")
    ax.plot(xs, [r[key50] for r in bl], "s-",  color=BL_C, linewidth=2, markersize=8,
            label=f"{BL_NAME} p50")
    ax.plot(xs, [r[key95] for r in bl], "s--", color=BL_C, linewidth=2, markersize=8,
            alpha=0.7, label=f"{BL_NAME} p95")
    ax.set_xlabel("Target request rate (rps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=8)
    ax.grid(True, alpha=0.3)


# Top-left: throughput (no percentile — single line each)
ax = axes[0][0]
ax.plot(xs, [r["tok_s"] for r in pd], "o-", color=PD_C, linewidth=2, markersize=8, label=PD_NAME)
ax.plot(xs, [r["tok_s"] for r in bl], "s-", color=BL_C, linewidth=2, markersize=8, label=BL_NAME)
ax.set_xlabel("Target request rate (rps)")
ax.set_ylabel("Output throughput (tokens/sec)")
ax.set_title("Output throughput")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

overlay(axes[0][1], "ttft_p50", "ttft_p95",
        "TTFT (s)", "Time-to-First-Token (p50 solid, p95 dashed)")
overlay(axes[1][0], "e2e_p50", "e2e_p95",
        "Request latency (s)", "End-to-end Request Latency (p50 solid, p95 dashed)")
overlay(axes[1][1], "itl_p50", "itl_p95",
        "ITL (ms)", "Inter-Token Latency (p50 solid, p95 dashed)")

plt.suptitle(
    "sarvam-30b on 1× MI325X (8 GPUs): 4P×TP1 + 4D×TP1 vs 8×TP1 baseline",
    y=1.00, fontsize=13,
)
plt.tight_layout()
out = ROOT / "comparison_4p4d_vs_baseline.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

#!/usr/bin/env python3
"""Plot TTFT and output tok/s for llm-d vs plain k8s on the 3-vendor MIXED
20-pod pool (4 NVIDIA H100-NVL + 8 AMD MI325X + 8 Intel Gaudi3) serving
ibm-granite/granite-4.1-8b.

k8s run was trimmed (warmup + rates 5,15,25,30) because rate=40 saturated
the round-robin pool and the harness timed out with no reports. llm-d ran
the full ladder 5..85.
"""
import pathlib
import matplotlib.pyplot as plt
import yaml

BASE = pathlib.Path(__file__).parent
LLMD = next(BASE.glob("results-llmd/*/inference-perf_*_llmd-mixed3v-granite"))
K8S = next(BASE.glob("results-k8s/*/inference-perf_*_k8s-mixed3v-granite-trim"))

# llmd stages: 0=warmup(30), 1=5, 2=15, 3=25, 4=40, 5=55, 6=70, 7=85
LLMD_STAGES = {1: 5, 2: 15, 3: 25, 4: 40, 5: 55, 6: 70, 7: 85}
# k8s stages (trim): 0=warmup(20), 1=5, 2=15, 3=25, 4=30
K8S_STAGES = {1: 5, 2: 15, 3: 25, 4: 30}


def load(root: pathlib.Path, stage: int) -> dict:
    p = next(root.glob(f"benchmark_report,_stage_{stage}_*.yaml"))
    with open(p) as f:
        return yaml.safe_load(f)


def extract(root: pathlib.Path, stages: dict) -> list[dict]:
    out = []
    for s, rate in stages.items():
        r = load(root, s)
        out.append({
            "rate": rate,
            "ttft": r["metrics"]["latency"]["time_to_first_token"]["mean"],
            "tok":  r["metrics"]["throughput"]["output_tokens_per_sec"],
        })
    return out


llmd_rows = extract(LLMD, LLMD_STAGES)
k8s_rows  = extract(K8S, K8S_STAGES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
title_sub = "Mixed pool: 4×H100-NVL + 8×MI325X + 8×Gaudi3 serving Granite-4.1-8B"

ax1.plot([r["rate"] for r in llmd_rows], [r["ttft"] for r in llmd_rows],
         "o-", label="llm-d (precise-prefix-cache)", linewidth=2, markersize=8)
ax1.plot([r["rate"] for r in k8s_rows],  [r["ttft"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8)
ax1.axvline(x=30, linestyle="--", color="grey", alpha=0.5)
ax1.text(31, 0.1, "k8s saturates\nbeyond rate 30\n(harness timeout)",
         fontsize=9, color="dimgrey", va="bottom")
ax1.set_xlabel("Target QPS (requests/sec)")
ax1.set_ylabel("Mean TTFT (seconds)")
ax1.set_title(f"Time To First Token vs QPS\n{title_sub}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

ax2.plot([r["rate"] for r in llmd_rows], [r["tok"] for r in llmd_rows],
         "o-", label="llm-d (precise-prefix-cache)", linewidth=2, markersize=8)
ax2.plot([r["rate"] for r in k8s_rows],  [r["tok"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8)
ax2.axvline(x=30, linestyle="--", color="grey", alpha=0.5)
ax2.text(31, 2000, "k8s saturates\nbeyond rate 30\n(harness timeout)",
         fontsize=9, color="dimgrey", va="bottom")
ax2.set_xlabel("Target QPS (requests/sec)")
ax2.set_ylabel("Output tokens/sec")
ax2.set_title(f"Output Throughput vs QPS\n{title_sub}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = BASE / "comparison_3vendor_pool.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'rate':<6} {'llmd_TTFT':<12} {'k8s_TTFT':<12} {'llmd_tok':<10} {'k8s_tok':<10}")
print("-" * 60)
k8s_map = {r["rate"]: r for r in k8s_rows}
for r in llmd_rows:
    k = k8s_map.get(r["rate"])
    k_ttft = f"{k['ttft']:.3f}" if k else "—"
    k_tok  = f"{k['tok']:.0f}"  if k else "— (timeout)"
    print(f"{r['rate']:<6} {r['ttft']:<12.3f} {k_ttft:<12} {r['tok']:<10.0f} {k_tok:<10}")

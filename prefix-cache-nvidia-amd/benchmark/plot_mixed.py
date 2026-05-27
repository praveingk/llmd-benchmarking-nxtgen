#!/usr/bin/env python3
"""Plot TTFT and output tok/s for llm-d vs plain k8s on the MIXED 12-pod pool
(4 NVIDIA H100 NVL + 8 AMD MI325X) serving ibm-granite/granite-4.1-8b.

k8s run was capped at rate 45 (the prior stage-6/rate=55 run hit harness
timeout). We use the overlapping stages (rates 5, 15, 25, 35, 45)."""
import pathlib
import matplotlib.pyplot as plt
import yaml

BASE = pathlib.Path(__file__).parent
LLMD = next(BASE.glob("results-llmd/*/inference-perf_*_llmd-mixed-granite"))
K8S = next(BASE.glob("results-k8s/*/inference-perf_*_k8s-mixed-granite"))
K8S_HIGH = next(BASE.glob("results-k8s-high/*/inference-perf_*_k8s-mixed-granite-high"))

# Stage 0 is warmup. llmd has 8 stages (rates 30,5,15,25,35,45,55,65).
LLMD_STAGES = {1: 5, 2: 15, 3: 25, 4: 35, 5: 45, 6: 55, 7: 65}
# k8s trimmed run produced rates 5-45 (stages 1-5).
K8S_STAGES = {1: 5, 2: 15, 3: 25, 4: 35, 5: 45}
# k8s-high focused run (warmup=25, then 55, 65) — stages 1 and 2 give 55 and 65.
K8S_HIGH_STAGES = {1: 55, 2: 65}


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
k8s_rows  = sorted(extract(K8S, K8S_STAGES) + extract(K8S_HIGH, K8S_HIGH_STAGES),
                   key=lambda r: r["rate"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot([r["rate"] for r in llmd_rows], [r["ttft"] for r in llmd_rows],
         "o-", label="llm-d (precise-prefix-cache)", linewidth=2, markersize=8)
ax1.plot([r["rate"] for r in k8s_rows],  [r["ttft"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8)
ax1.set_xlabel("Target QPS (requests/sec)")
ax1.set_ylabel("Mean TTFT (seconds)")
ax1.set_title("Time To First Token vs QPS\nMixed pool: 4×H100-NVL + 8×MI325X serving Granite-4.1-8B")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

ax2.plot([r["rate"] for r in llmd_rows], [r["tok"] for r in llmd_rows],
         "o-", label="llm-d (precise-prefix-cache)", linewidth=2, markersize=8)
ax2.plot([r["rate"] for r in k8s_rows],  [r["tok"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8)
ax2.set_xlabel("Target QPS (requests/sec)")
ax2.set_ylabel("Output tokens/sec")
ax2.set_title("Output Throughput vs QPS\nMixed pool: 4×H100-NVL + 8×MI325X serving Granite-4.1-8B")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = BASE / "comparison_mixed_pool.png"
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

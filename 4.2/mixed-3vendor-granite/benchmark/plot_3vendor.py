#!/usr/bin/env python3
"""Plot TTFT and output tok/s for llm-d vs plain k8s on the 3-vendor mixed
20-pod pool (4 H100-NVL + 8 MI325X + 8 Gaudi3) serving granite-4.1-8b.

Both runs cover rates 5..85. k8s saturates at rate 25-40 (~9.5K peak), then
throughput DECLINES to 7.5K at rate 85 as queue depth blows up; TTFT hits 36s
under the same load. llm-d keeps scaling cleanly to 14.2K @ rate 85.
"""
import json
import pathlib

import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent
LLMD = next(BASE.glob("results-llmd/*/inference-perf_*_llmd-3vendor-granite"))
K8S  = next(BASE.glob("results-k8s/*/inference-perf_*_k8s-3vendor-granite-trim"))

# Stage indices to (rate qps) — stage 0 is warmup, omit from curves
LLMD_STAGES = {1: 5, 2: 15, 3: 25, 4: 40, 5: 55, 6: 70, 7: 85}
K8S_STAGES  = {1: 5, 2: 15, 3: 25, 4: 40, 5: 55, 6: 70, 7: 85}


def load_stage(root: pathlib.Path, stage: int) -> dict:
    with open(root / f"stage_{stage}_lifecycle_metrics.json") as f:
        return json.load(f)


def extract(root: pathlib.Path, stages: dict) -> list[dict]:
    out = []
    for s, rate in stages.items():
        d = load_stage(root, s)
        succ = d.get("successes", {})
        out.append({
            "rate": rate,
            "ttft": succ.get("latency", {}).get("time_to_first_token", {}).get("mean", float("nan")),
            "tok":  succ.get("throughput", {}).get("output_tokens_per_sec", float("nan")),
            "ok":   succ.get("count", 0),
            "fail": d.get("failures", {}).get("count", 0),
        })
    return out


llmd_rows = extract(LLMD, LLMD_STAGES)
k8s_rows  = extract(K8S, K8S_STAGES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
title_sub = "Mixed pool: 4×H100-NVL + 8×MI325X + 8×Gaudi3 serving Granite-4.1-8B"

# --- TTFT panel ---
ax1.plot([r["rate"] for r in llmd_rows], [r["ttft"] for r in llmd_rows],
         "o-", label="llm-d (prefix-cache-aware EPP)", linewidth=2, markersize=8)
ax1.plot([r["rate"] for r in k8s_rows],  [r["ttft"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8, color="tab:orange")
ax1.set_xlabel("Target QPS (requests/sec)")
ax1.set_ylabel("Mean TTFT (seconds)")
ax1.set_title(f"Time To First Token vs QPS\n{title_sub}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

# --- Throughput panel ---
ax2.plot([r["rate"] for r in llmd_rows], [r["tok"] for r in llmd_rows],
         "o-", label="llm-d (prefix-cache-aware EPP)", linewidth=2, markersize=8)
ax2.plot([r["rate"] for r in k8s_rows],  [r["tok"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8, color="tab:orange")
ax2.set_xlabel("Target QPS (requests/sec)")
ax2.set_ylabel("Output tokens/sec")
ax2.set_title(f"Output Throughput vs QPS\n{title_sub}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = BASE / "comparison_3vendor.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

print()
print(f"{'rate':<6} {'llmd_tok':<10} {'k8s_tok':<14} {'llmd_TTFT':<12} {'k8s_TTFT':<14} {'llmd_ok/fail':<14} {'k8s_ok/fail':<12}")
print("-" * 100)
k8s_map = {r["rate"]: r for r in k8s_rows}
for r in llmd_rows:
    k = k8s_map.get(r["rate"])
    k_tok  = f"{k['tok']:.0f}"  if k else "— (saturated)"
    k_ttft = f"{k['ttft']:.3f}" if k else "—"
    k_okf  = f"{k['ok']}/{k['fail']}" if k else "—"
    print(f"{r['rate']:<6} {r['tok']:<10.0f} {k_tok:<14} "
          f"{r['ttft']:<12.3f} {k_ttft:<14} "
          f"{r['ok']}/{r['fail']:<10} {k_okf:<12}")

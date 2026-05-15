#!/usr/bin/env python3
"""Plot TTFT and output tok/s for llm-d vs plain k8s on the gaudi-only 8-pod
pool (8 Intel Gaudi3) serving ibm-granite/granite-4.1-8b.

llm-d ran the full ladder rates 3..50. k8s was trimmed at rate 25 because
beyond saturation the harness/API would hit DNS timeouts during the long
queueing stages. Stage 4 (rate 25) on k8s already shows pool saturation:
TTFT jumped 18x while throughput plateaued.
"""
import pathlib
import json
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent
LLMD = next(BASE.glob("results-llmd-bs128-ms256/*/inference-perf_*_llmd-gaudi-only-granite"))
K8S = next(BASE.glob("results-k8s/*/inference-perf_*_k8s-gaudi-only-granite-trim"))

# Stage indices to (rate qps) mapping. Stage 0 is warmup; we omit it from the curve.
LLMD_STAGES = {1: 3, 2: 8, 3: 15, 4: 25, 5: 35, 6: 50}
K8S_STAGES  = {1: 3, 2: 8, 3: 15, 4: 25}


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
k8s_rows  = extract(K8S,  K8S_STAGES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "8×Gaudi3 serving Granite-4.1-8B  (block-size=128, max-num-seqs=256)",
    fontsize=12, y=1.02,
)

# --- TTFT panel ---
ax1.plot([r["rate"] for r in llmd_rows], [r["ttft"] for r in llmd_rows],
         "o-", label="llm-d (prefix-cache-aware EPP)", linewidth=2, markersize=8)
ax1.plot([r["rate"] for r in k8s_rows],  [r["ttft"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8, color="tab:orange")
ax1.axvline(x=25, linestyle="--", color="grey", alpha=0.5)
ax1.annotate(
    "k8s saturates at rate 25\nTTFT jumps 18× (1.5s → 27s)\nrates >25 queue indefinitely",
    xy=(25, 27.5), xytext=(28, 5.0),
    fontsize=9, color="dimgrey",
    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1, alpha=0.6),
)
ax1.set_xlabel("Target QPS (requests/sec)")
ax1.set_ylabel("Mean TTFT (seconds)")
ax1.set_title("Time To First Token vs QPS")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")
ax1.set_xlim(0, 55)

# --- Throughput panel ---
ax2.plot([r["rate"] for r in llmd_rows], [r["tok"] for r in llmd_rows],
         "o-", label="llm-d (prefix-cache-aware EPP)", linewidth=2, markersize=8)
ax2.plot([r["rate"] for r in k8s_rows],  [r["tok"] for r in k8s_rows],
         "s-", label="plain k8s (round-robin)", linewidth=2, markersize=8, color="tab:orange")
ax2.axvline(x=25, linestyle="--", color="grey", alpha=0.5)
ax2.annotate(
    "k8s plateaus at ~3.4K tok/s\nllm-d still scaling: 5.0K @ rate 50",
    xy=(25, 3399), xytext=(28, 2500),
    fontsize=9, color="dimgrey",
    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1, alpha=0.6),
)
ax2.set_xlabel("Target QPS (requests/sec)")
ax2.set_ylabel("Output tokens/sec")
ax2.set_title("Output Throughput vs QPS")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 55)
ax2.set_ylim(1000, 5500)

plt.tight_layout()
out = BASE / "comparison_gaudi_only.png"
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

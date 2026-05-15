# 4.2 / Gaudi-only

Single-vendor inference pool of **8 Intel Gaudi3 cards** serving `ibm-granite/granite-4.1-8b` under the precise-prefix-cache-aware llm-d guide. Compares llm-d (prefix-cache-aware EPP routing) vs plain k8s (`ClusterIP` round-robin) on the same backend.

## Why this exists

The 4.2 3-vendor experiment (`../`) added 8 Gaudi3 pods to the H100+AMD pool. The Gaudi pods dragged total throughput down (~7K out tok/s on 20 mixed pods) and warmup was unstable. This directory rebuilds the gaudi pods standalone with v0.7.0 image and a tuned flag set, then runs the same benchmark for a fair comparison.

## Result

![llm-d vs k8s on 8 Gaudi3](benchmark/comparison_gaudi_only.png)

- **Peak: 5,011 out tok/s** at rate=50 with llm-d (vs 1,664 the original transplanted config got).
- **At saturation (rate=25), llm-d delivers +34% throughput and 18× better TTFT** than plain k8s round-robin.
- k8s round-robin saturates at rate=25 (TTFT 1.5s → 27s, tok/s plateaus at 3.4K). Beyond that the queue grows unbounded and the harness times out.
- Per-pod Gaudi3: 626 out tok/s. Still ~2.4–3× behind AMD MI325X (~2000) and H100 (~1500) on this workload — likely vllm-gaudi's hybrid-Mamba kernel maturity, not a flag-tuning issue.

## What was tuned (and why)

The original 4.2 mixed values.yaml gave **1,664 tok/s** when transplanted as gaudi-only with v0.7.0. Three changes raised that to **5,011 tok/s** (3.0×):

| change | rationale | effect |
|---|---|---|
| `--block-size 16 → 128` | HPU dispatch overhead per block dominates per-token latency at small block sizes | ITL 211ms → 89ms, peak 1664 → 4625 |
| `--max-num-seqs 64 → 256` | bs=64 left ~10 GiB graph reservation unused; more concurrent decodes | peak 4625 → 5011, TTFT@50qps 93s → 2.8s |
| Drop `VLLM_*_BUCKET_*` envs | v0.7.0 uses exponential bucketing; these envs are warned-and-ignored | (no effect; cleanup) |

The bucket grid is now driven by the engine flags (`max-model-len 9216`, `max-num-seqs 256`, `max-num-batched-tokens 16384`, `gpu-memory-utilization 0.75`) — see [`ms-kv-events/values.yaml`](ms-kv-events/values.yaml).

## Deploy

```bash
# First time: HF_TOKEN env required, or have llm-d-hf-token secret in $SOURCE_NAMESPACE
HF_TOKEN=hf_xxx ./deploy.sh deploy

# Subsequent
./deploy.sh deploy        # idempotent
./deploy.sh status        # pod / svc / release listing
./deploy.sh test          # 3 sanity completions through the standalone EPP
./deploy.sh destroy       # remove releases (keep namespace + token)
```

The script:
1. Creates namespace `llm-d-granite-kv` and copies `llm-d-hf-token` from `$SOURCE_NAMESPACE` if needed.
2. Helmfile-applies `infra-granite-gaudi`, `gaie-granite-gaudi`, `ms-granite-gaudi`.
3. Installs the standalone EPP chart (`precise-granite-gaudi`).
4. Applies `httproute.yaml` (decode-backend Service + InferencePool v1alpha2 + HTTPRoute).
5. Waits up to 60 min for 8/8 decode pods Ready (warmup is ~85-110s/pod, but they contend during init so 30-45 min wall-clock for all 8 is normal).

## Benchmark

```bash
cd benchmark
./run_only.sh -c config-llmd.yaml -o results-llmd-bs128-ms256
./run_only.sh -c config-k8s-trim.yaml  -o results-k8s
python3 plot_gaudi_only.py
```

`config-llmd.yaml` ramps rates 3→50; `config-k8s-trim.yaml` stops at 25 because beyond saturation the round-robin pool queues for 10+ minutes per stage and the harness's `kubectl exec` connection becomes a fragile dependency.

For the round-robin path, a `decode-clusterip` Service is created at deploy time:

```yaml
apiVersion: v1
kind: Service
metadata: { name: decode-clusterip, namespace: llm-d-granite-kv }
spec:
  type: ClusterIP
  selector:
    llm-d.ai/inference-serving: "true"
    llm-d.ai/guide: precise-prefix-cache-aware
    llm-d.ai/role: decode
  ports: [{ name: vllm, port: 8000, targetPort: 8000 }]
```

## Gotchas captured during this run

- **`VLLM_BUILD=1.23.0.695` pin is required** even with v0.7.0 — same Habana SDK underneath, same auto-detect failure under SCC-locked `HOME`.
- **Switch deployment strategy to `Recreate`** before flag changes — the default RollingUpdate maxSurge tries to spawn 2 surge pods, but all 8 Gaudi cards on the node are taken, so they sit `Pending` indefinitely.
- **EPP must use approximate `prefix-cache-scorer`** — the inference-scheduler v0.8.0 image is built without `embedded_tokenizers`, so `precise-prefix-cache-scorer` (which needs an HF tokenizer) crashes at startup. Approximate hash-based routing still gives the +34% / 18× wins shown above.
- **Bump gaie chart EPP image to `v0.8.0`** in `gaie-kv-events/values.yaml`. Chart default is `v0.7.1` which has stricter tokenizer requirements and crashes differently.
- **Kubernetes API DNS flakes (`api.<cluster>:6443`) kill the harness** if the run takes more than ~30 min. The trimmed k8s config (rates 3→25) finishes in ~5 min and avoids this.

## Layout

```
4.2/gaudi-only/
├── README.md                     # this file
├── deploy.sh                     # entry point
├── helmfile.yaml.gotmpl          # only deploys infra + gaie + ms-gaudi
├── httproute.yaml                # InferencePool v1alpha2 + HTTPRoute
├── prereq/                       # bundled istio gateway env values
├── smoke-warmup.yaml             # single-pod manifest used during smoke validation
├── ms-kv-events/values.yaml      # MS chart values (vllm pod spec)
├── gaie-kv-events/values.yaml    # GAIE inferencepool chart values (gaie EPP)
├── standalone-values/values.yaml # standalone EPP chart values (router/scorer)
└── benchmark/
    ├── config-llmd.yaml          # rates 3..50
    ├── config-k8s.yaml           # rates 3..50 (saturates past 25; use trim instead)
    ├── config-k8s-trim.yaml      # rates 3..25 — used for the comparison
    ├── run_only.sh               # llm-d-benchmark harness launcher (copied from 4.2)
    ├── plot_gaudi_only.py        # generates the comparison PNG
    └── results-llmd-bs128-ms256/
    └── results-k8s/
    └── comparison_gaudi_only.png
```

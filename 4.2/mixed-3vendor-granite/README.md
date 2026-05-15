# 4.2 / Mixed 3-Vendor Granite-4.1-8b

3-vendor inference pool of **4× H100-NVL + 8× MI325X + 8× Gaudi3 (= 20 pods)** serving `ibm-granite/granite-4.1-8b` under the precise-prefix-cache-aware llm-d guide. Compares llm-d (prefix-cache-aware EPP routing) vs plain k8s round-robin on the same backend.

## Why this exists

The original [4.2/](../) experiment ran the same 3-vendor mix with `v0.6.0` Gaudi image and the (then) un-tuned bucket grid; the Gaudi pods dragged peak throughput down to ~7K tok/s and were unstable under warmup. After the [`gaudi-only/`](../gaudi-only/) experiment validated `v0.7.0` + `block-size 128 + max-num-seqs 256` lifts Gaudi3 to 5K tok/s alone, this directory rebuilds the 3-vendor mix with the optimized Gaudi config so the pool comparison is fair.

## Layout

```
4.2/mixed-3vendor-granite/
├── README.md                         # this file
├── helmfile.yaml.gotmpl              # 4 releases: infra + gaie + ms-{nvidia,amd,gaudi}
├── deploy.sh                         # entry point (DEPLOY_AMD=true to enable AMD release)
├── httproute.yaml                    # decode-backend + decode-clusterip + InferencePool + HTTPRoute
├── downloader-gaudi.yaml             # one-shot pod that pre-stages granite to gaudi node hostPath cache
├── prereq/                           # bundled istio gateway env values
├── ms-nvidia/values.yaml             # 4× H100-NVL, hostPath HF cache, block-size 16, gpu-mem-util 0.85
├── ms-amd/values.yaml                # 8× MI325X, mountModelVolume=true, block-size 16, gpu-mem-util 0.85
├── ms-gaudi/values.yaml              # 8× Gaudi3, hostPath HF cache, OPTIMIZED config from gaudi-only:
│                                     #   block-size 128, max-num-seqs 256, gpu-mem-util 0.75,
│                                     #   max-model-len 9216, VLLM_BUILD pin, exponential bucketing
├── gaie-kv-events/values.yaml        # GAIE inferencepool chart values (approximate prefix-cache-scorer)
├── standalone-values/values.yaml     # standalone EPP chart values (approximate scorer)
└── benchmark/
    ├── config-llmd-mixed.yaml        # rates 3..50 (same as gaudi-only for direct comparison)
    ├── config-k8s-trim-mixed.yaml    # rates 3..25 (round-robin saturates earlier)
    ├── run_only.sh                   # llm-d-benchmark harness launcher
    └── plot_gaudi_only.py            # leftover from cp; rename or replace for 3-vendor plot
```

## Pre-stage the model (do once before first deploy)

The chart's `mountModelVolume: true` would have all 8 Gaudi pods (and 4 NVIDIA pods) download the model concurrently. With granite at ~17 GiB this isn't catastrophic but **using a node-local hostPath HF cache is much faster + simpler**.

- **NVIDIA H100-NVL nodes**: `/var/lib/llm-d-hf-cache` is **already populated** from prior 4.1/4.2 work (granite + sarvam blobs already there, SELinux context set).
- **Gaudi3 node**: pre-stage granite once via the included downloader pod:
  ```bash
  kubectl apply -f downloader-gaudi.yaml         # ~4 min for 17 GiB
  kubectl wait pod/granite-hf-downloader-gaudi -n llm-d-granite-kv --for=condition=Ready=False --timeout=10m
  kubectl logs granite-hf-downloader-gaudi -n llm-d-granite-kv --tail=5
  kubectl delete pod granite-hf-downloader-gaudi -n llm-d-granite-kv --grace-period=0 --force
  ```
- **AMD MI325X node**: not yet pre-staged. AMD ms values use `mountModelVolume: true` so the chart will download lazily when the deploy lands. Once AMD nodes are available, you can switch to hostPath if desired (mirrors `ms-gaudi/values.yaml` patterns).

The hostPath path is `/var/lib/llm-d-hf-cache`; SELinux context required: `chcon -R -t container_file_t /var/lib/llm-d-hf-cache` on each node (already set on existing nodes).

## Deploy

```bash
# Phase 1: NVIDIA + Gaudi only (AMD nodes not yet allocated)
HF_TOKEN=hf_xxx ./deploy.sh deploy        # ~12 min for 4 NVIDIA pods + 30-40 min for 8 Gaudi pods

# Phase 2: when AMD nodes arrive
./deploy.sh deploy-amd                    # adds 8 MI325X pods; EPP picks them up via shared label selector

# Verify routing across all up vendors
./deploy.sh test                          # 3 sanity completions through the standalone EPP
./deploy.sh status                        # pod / svc / release listing, broken down by vendor
```

The script is idempotent — `deploy` skips AMD by default; `deploy-amd` flips the helmfile gate. Either way, the helm releases are merged into the existing infra + gaie + EPP infrastructure.

## Benchmark

```bash
cd benchmark
./run_only.sh -c config-llmd-mixed.yaml      -o results-llmd-mixed
./run_only.sh -c config-k8s-trim-mixed.yaml  -o results-k8s-mixed
# adapt plot_gaudi_only.py to plot 3-vendor results, then:
# python3 plot_3vendor.py
```

The benchmark base_url for llm-d points at `precise-granite-mixed-epp.llm-d-granite-mixed-kv.svc.cluster.local:8081` (standalone EPP); the k8s round-robin run points at `decode-clusterip` Service which load-balances across all decode pods regardless of vendor.

## Why a separate namespace from gaudi-only

The previous [`../gaudi-only/`](../gaudi-only/) experiment lives in `llm-d-granite-kv`. To preserve those clean single-vendor numbers (5,011 out tok/s peak), the mixed pool runs in **`llm-d-granite-mixed-kv`** with its own infra/gaie/EPP. Both namespaces can coexist (gaudi-only is scaled to 0 while mixed is running, so no GPU contention).

## Headline numbers (TBD — pending AMD deployment)

Will be populated after Phase 2. Expected behavior based on [4.1 mixed (NVIDIA+AMD only) sarvam-30b](../../4.1/precise-prefix-sarvam-30b-mixed/):

- llm-d wins much bigger on heterogeneous pools because round-robin punishes the slowest vendor.
- Per-vendor pod throughput on this workload (from earlier runs):
  - **AMD MI325X**: ~2,000 out tok/s/pod
  - **NVIDIA H100-NVL**: ~1,500 out tok/s/pod
  - **Gaudi3 (optimized)**: ~625 out tok/s/pod
- Total pool ceiling estimate: 8 × 2K + 4 × 1.5K + 8 × 625 = **~27 K out tok/s** for llm-d.
- k8s round-robin should saturate much lower because slow Gaudi pods become queueing sinks.

## Gotchas captured in earlier work

These all apply here too — reusing the patterns:

- **Use approximate `prefix-cache-scorer`** in both EPPs. The v0.8.0 inference-scheduler image is built without `embedded_tokenizers`; precise scorer fails at startup. Same workaround as 4.1/4.2 + gaudi-only.
- **Bump gaie-EPP image to `v0.8.0`** in `gaie-kv-events/values.yaml`. Chart default `v0.7.1` has stricter tokenizer requirements and crashes differently.
- **Patch deployment strategy to `Recreate`** before flag changes — the default RollingUpdate maxSurge tries to spawn surge pods, but on the gaudi node all 8 cards are taken so they sit Pending. Runs fine but rollouts thrash.
- **Gaudi `VLLM_BUILD=1.23.0.695` pin still required** even with v0.7.0 — same Habana SDK underneath, same auto-detect failure under SCC-locked `HOME`.
- **Kubernetes API DNS flakes (`api.<cluster>:6443`) kill long-running benchmarks** if the run takes more than ~30 min. Trim k8s rate ladder to stop before saturation tail.

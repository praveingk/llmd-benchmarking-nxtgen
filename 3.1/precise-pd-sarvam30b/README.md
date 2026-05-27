# P/D Disaggregation — sarvam-30b on 8× AMD MI325X (4P + 4D)

llm-d v0.0.7 prefill/decode disaggregation on a single 8-GPU MI325X node, compared
against a plain Kubernetes baseline of 8 monolithic vLLM pods (TP=1) behind a
ClusterIP service. KV cache is transferred from prefill to decode pods over NIXL
on the shared 100 G RoCE network.

## Layout

```
.
├── helmfile.yaml.gotmpl                # llm-d stack (infra + gaie + ms)
├── httproute.yaml                      # Gateway-API HTTPRoute
├── gaie-pd/values.yaml                 # InferencePool / EPP (precise-prefix scorer)
├── ms-pd/
│   ├── values_sarvam_pd_4p4d_amd.yaml  # 4 prefill + 4 decode (TP=1 each)
│   └── baseline_amd_8tp1.yaml          # 8× monolithic vLLM (TP=1) for round-robin baseline
└── benchmark/4p4d-amd-prefill-heavy/
    ├── results-llmd-pd/                # PD run (4P+4D)
    ├── results-baseline/               # k8s round-robin run (8×TP1)
    ├── comparison_4p4d_vs_baseline.png # 2×2 plot: throughput, TTFT p50, E2E p50, ITL p50
    ├── plot_4p4d_vs_baseline.py
    ├── config-llmd-pd.yaml             # inference-perf config (PD)
    ├── config-baseline.yaml            # inference-perf config (baseline)
    ├── bench-llmd-pd.log
    ├── bench-baseline.log
    └── run_only.sh
```

## Workload

inference-perf `shared_prefix_synthetic`, prefill-heavy:

- system prompt = 6 000 tokens (shared across requests)
- question     = 2 000 tokens
- output       = 500 tokens
- rate ladder  = 1, 5, 10, 25, 50, 75, 100, 125, 150 req/s

## Deploy llm-d (4P + 4D)

```bash
export NAMESPACE=llm-d-pd
export RELEASE_NAME_POSTFIX=pd
helmfile -e amd_sarvam_4p4d apply
kubectl apply -f httproute.yaml
```

## Deploy baseline (8× TP=1, round-robin)

```bash
kubectl apply -f ms-pd/baseline_amd_8tp1.yaml
```

## Run benchmarks

```bash
cd benchmark/4p4d-amd-prefill-heavy
./run_only.sh
python3 plot_4p4d_vs_baseline.py
```

## Headline result

At rate 100 req/s (saturating load), prefill-heavy workload:

| Config                       | Throughput (out tok/s) | TTFT p50 | ITL p50 | E2E p50 |
| ---                          | ---                    | ---      | ---     | ---     |
| 8× TP=1 baseline (k8s RR)    | 24 028                 | 4.79 s   | 114 ms  | 113 s   |
| 4P + 4D (llm-d PD)           | 21 583                 | 0.59 s   | 81 ms   | 59 s    |

PD trades ~12% peak throughput for ~8× lower TTFT and ~30–50% lower ITL under
load, because role isolation eliminates prefill/decode interference on the same
GPU. PD also held zero failures across 30 K requests; the baseline began
dropping requests at higher rates.

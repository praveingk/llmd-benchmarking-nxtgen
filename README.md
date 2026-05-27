# llmd-benchmarking

Deployment configurations and benchmark results for llm-d on a heterogeneous
GPU cluster (NxtGen sovereign cloud — NVIDIA H100-NVL + AMD MI325X +
Intel Gaudi3, all on a shared 100 G RoCE fabric).

The headline question across these experiments: **can a single Kubernetes-native
serving layer take a heterogeneous fleet of accelerators and beat plain k8s
round-robin on throughput and TTFT, with no application-level changes?** The
answer is yes — see [`Results.md`](Results.md) for the full result matrix.

## Experiments

| Folder | Technique | Hardware | Highlight |
| --- | --- | --- | --- |
| [`prefix-cache-nvidia`](prefix-cache-nvidia/) | Precise prefix-cache-aware EPP | 4× NVIDIA H100-NVL | granite +25–36% tput / 16× TTFT, sarvam 2× tput / 22× TTFT vs k8s |
| [`prefix-cache-amd`](prefix-cache-amd/) | Precise prefix-cache-aware EPP | 8× AMD MI325X | granite +79% tput / 21× TTFT, sarvam +85% tput / 5× TTFT vs k8s |
| [`prefix-cache-gaudi`](prefix-cache-gaudi/) | Precise prefix-cache-aware EPP | 8× Intel Gaudi3 | granite +34% tput / 18× TTFT vs k8s (sarvam-30b on Gaudi was a dead-end) |
| [`prefix-cache-nvidia-amd`](prefix-cache-nvidia-amd/) | Precise prefix-cache-aware EPP | 12 GPUs (4 H100 + 8 MI325X) | granite +85% tput / 3.4–5.6× TTFT, sarvam ~3× tput / 2.85–4.54× TTFT |
| [`prefix-cache-3vendor`](prefix-cache-3vendor/) | Precise prefix-cache-aware EPP | 20 GPUs (4 H100 + 8 MI325X + 8 Gaudi3) | granite **+91% tput / 5.4× TTFT** vs k8s — biggest llm-d win |
| [`pd-disaggregation`](pd-disaggregation/) | Prefill/Decode disaggregation (NIXL over RoCE) | 8× AMD MI325X (4P + 4D) | sarvam **8× lower TTFT, 30–50% lower ITL** at saturation, 12% peak-tput give-back |

Each experiment folder is self-contained: helmfile + values for the llm-d
stack, a baseline manifest for the plain-k8s comparison, the
inference-perf benchmark configs, raw `stage_*_lifecycle_metrics.json` results
for both runs, and the plot script that produces the comparison chart.

## Models served

- `ibm-granite/granite-4.1-8b` — 8 B parameter, hybrid-Mamba transformer
- `sarvamai/sarvam-30b` — 30 B MoE, Indic-multilingual model with custom vLLM kernels

## Workload

Prefill-heavy `shared_prefix_synthetic` from
[inference-perf](https://github.com/kubernetes-sigs/inference-perf): a long
shared system prompt + short question + decode-tolerant output. This matches
production RAG, chat, and citizen-services traffic profiles where prefix-cache
routing has the most room to win.

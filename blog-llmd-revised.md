---
title: "Heterogeneous Inference at Scale on a 3-Vendor Sovereign Cluster"
---

# Heterogeneous Inference at Scale on a 3-Vendor Sovereign Cluster

*Pravein Govindan Kannan ([pravein.govindan.kannan@ibm.com](mailto:pravein.govindan.kannan@ibm.com)), Praveen Jayachandran ([praveen.j@in.ibm.com](mailto:praveen.j@in.ibm.com)), Jaikrishnan Hari ([jaikhari@in.ibm.com](mailto:jaikhari@in.ibm.com))*
*Prasad Mukhedkar ([pmukhedk@redhat.com](mailto:pmukhedk@redhat.com)), Varun Raste ([varun.raste@ibm.com](mailto:varun.raste@ibm.com)), Vinod Pathangay ([vpathang@redhat.com](mailto:vpathang@redhat.com))*
*Jayanth Babu Reddy ([jayanth.reddy@nxtgen.ai](mailto:jayanth.reddy@nxtgen.ai)), Abhisyant Anasapurapu ([abhisyant@nxtgen.ai](mailto:abhisyant@nxtgen.ai))*

Most production inference clusters today are single-vendor — not because that is the optimal design, but because making a multi-vendor cluster actually work has been hard enough that organizations have lived with the lock-in. That equilibrium is breaking down. Procurement cycles bring new generations alongside older ones, supply constraints push teams across vendors, and the cost gap between premium and lower-tier accelerators makes a one-size-fits-all fleet increasingly expensive to defend. The result is that real production fleets are accumulating heterogeneity whether or not the architecture planned for it. Treated as a first-class concern, that heterogeneity unlocks real value: lower-cost accelerators absorb prefill-heavy or batch-tolerant workloads, premium hardware is reserved for latency-sensitive paths, stranded capacity gets reclaimed, and the organization stops being hostage to a single supplier's roadmap or pricing. The case is sharper still for sovereign and on-premise deployments, where data residency, regulatory alignment, and the long-term economics of high-volume inference are pushing AI workloads off centralized hyperscaler stacks.

Making it work is hard in practice. Driver stacks, firmware versions, container runtimes, and hardware-specific attention kernels diverge across vendors, and there is no standardized performance contract to lean on. **vLLM and llm-d together close this gap** — vLLM as a high-performance inference engine with broad accelerator support, and **llm-d** as a Kubernetes-native distributed serving layer that brings intelligent request routing and prefill/decode disaggregation to heterogeneous clusters.

**IBM Research, Red Hat, and NxtGen Cloud Technologies** — one of India's leading sovereign cloud providers — came together to put this thesis under pressure with a first-of-a-kind proof-of-concept: distributed inference across a heterogeneous fleet, on a single coherent serving layer, with no application-level changes.

## At a glance

- **20-pod 3-vendor pool (4× H100-NVL + 8× MI325X + 8× Gaudi3), `granite-4.1-8b`, prefill-heavy traffic:** llm-d delivers **14.2 K out tok/s** vs **9.6 K** for plain k8s round-robin (**+91% throughput**), and **TTFT 6.8 s vs 36.4 s (5.4× better)** at peak load.
- **The prefix-cache advantage scales with pool heterogeneity:** single-vendor pools see +25–85% throughput; the 3-vendor mix is where llm-d wins biggest, because round-robin is most punished by capacity mismatches between vendors.
- **Prefill/Decode disaggregation** on a single 8-GPU MI325X node delivers **4× lower TTFT** at saturation and **zero failures across 30 K requests**, with peak throughput within 12% of the monolithic baseline — the right call for interactive workloads where TTFT is the SLO that matters.

## Setup

We performed experiments on the **NxtGen sovereign cloud**, with the following accelerator pools available within a single OpenShift AI cluster:

| Pool | Hardware | Count |
| --- | --- | --- |
| NVIDIA | H100-NVL (2 nodes × 2 GPUs) | 4 |
| AMD | MI325X (1 node) | 8 |
| Intel | Gaudi3 (1 node) | 8 |

All nodes are connected over a shared **100 G RoCE** network. We pinned each vLLM replica to a single accelerator card (TP = 1) to maximize the number of independent serving instances and exercise the routing layer.

Models served:

- `ibm-granite/granite-4.1-8b` (8 B parameter, hybrid-Mamba transformer)
- `sarvamai/sarvam-30b` (30 B MoE, Indic-multilingual model with custom vLLM kernels)

## Prefix-aware Caching

Many production LLM workloads — RAG, chat, retrieval-augmented assistants — share significant prompt structure across requests, often re-using long system prompts or conversation histories. When a new request begins with a prefix that was already computed for an earlier request, the resulting KV cache can be reused directly instead of recomputed. **Prefix-cache-aware routing** sends each request to the vLLM instance most likely to already hold its prefix. The challenge in a heterogeneous cluster is knowing where that cache lives in real time; llm-d's scheduler tracks every vLLM instance's KV state via live KV events to make that decision, and is indifferent to whether the best-matched instance runs on H100, MI325X, or Gaudi3.

We deployed llm-d v0.0.7 with precise-prefix-aware caching using the [well-lit path guide](https://github.com/llm-d/llm-d/tree/main/guides/precise-prefix-cache-aware). Each vendor's pods are deployed as a separate Helm release in the same namespace; only the `nodeSelector` and a small set of vendor-specific tuning flags (e.g. Gaudi's `--block-size 128`, `--max-num-seqs 256`, `VLLM_BUILD` pin) vary between releases. All pods carry the same selector labels and register with a single `InferencePool` maintained by llm-d's router. For the baseline, we use a `ClusterIP` service over the same set of pods to drive plain Kubernetes round-robin scheduling — same pods, same vLLM, same flags; only the routing layer differs.

Across every pool we tested — single-vendor (NVIDIA-only / AMD-only / Gaudi-only) and heterogeneous (NVIDIA+AMD, NVIDIA+AMD+Gaudi) — **llm-d's prefix-cache-aware routing consistently wins over plain k8s round-robin on prefill-heavy workloads** (long shared system prompt + short question, decode-tolerant output) on both throughput and TTFT. The advantage grows with pool size and heterogeneity:

| Pool | Pods | Model | Throughput edge (llm-d vs k8s) | TTFT edge |
| --- | --- | --- | --- | --- |
| NVIDIA-only | 4 H100-NVL | granite-4.1-8b | +25–36% | 16× |
| NVIDIA-only | 4 H100-NVL | sarvam-30b | 2× | 22× |
| AMD-only | 8 MI325X | granite-4.1-8b | +79% | 21× |
| AMD-only | 8 MI325X | sarvam-30b | +85% (29 K vs 17 K out tok/s) | 5× |
| Gaudi-only | 8 Gaudi3 | granite-4.1-8b | +34% | 18× |
| NVIDIA + AMD | 12 | granite-4.1-8b | +85% (19.4 K vs 10–11 K) | 3.4–5.6× |
| NVIDIA + AMD | 12 | sarvam-30b | ~3× @ 200 qps | 2.85–4.54× |
| **NVIDIA + AMD + Gaudi** | **20** | **granite-4.1-8b** | **+91% @ 85 qps** | **5.4×** |

**Why llm-d wins biggest on heterogeneous pools:** k8s round-robin spreads requests evenly regardless of pod speed, so a single slow vendor becomes a queueing sink that drags total throughput down. llm-d's prefix-cache-aware EPP routes around saturated pods and concentrates cache hits on warm ones — heterogeneity stops being a penalty.

### Single-vendor pools — granite-4.1-8b, prefill-heavy (~7.2k ISL + 1k OSL)

We start with the per-vendor baselines so the heterogeneous results below have a reference point.

**4× NVIDIA H100-NVL.** llm-d improves TTFT by up to 16× compared to k8s, and output throughput by 25–36%.

![](prefix-cache-nvidia/precise-prefix-granite/benchmark-results/comparison_rates_3_to_22%20copy.png)

**8× AMD MI325X.** llm-d delivers up to 21× better TTFT and +79% throughput vs k8s round-robin on this AMD-only granite deployment.

![](prefix-cache-amd/precise-prefix-granite-amd-fresh/benchmark/comparison_amd_fresh.png)

**8× Intel Gaudi3.** At saturation (rate 25), llm-d delivers +34% throughput and ~18× better TTFT vs plain k8s round-robin.

![](prefix-cache-gaudi/granite-8b/benchmark/comparison_gaudi_only.png)

### Single-vendor pools — sarvam-30b (multilingual MoE, prefill-heavy)

**4× NVIDIA H100-NVL on sarvam-30b.** llm-d delivers 2× the throughput and 22× better TTFT. k8s saturates around rate 25–30; llm-d keeps scaling.

![](prefix-cache-nvidia/precise-prefix-sarvam-30b/benchmark/comparison_rates_3_to_50.png)

**8× AMD MI325X on sarvam-30b.** While k8s throughput plateaus at 15–17 K out tok/s, llm-d goes up to 29 K — 85% higher throughput. TTFT-wise llm-d is up to 5× faster at lower rates.

![](prefix-cache-amd/precise-prefix-sarvam-30b-amd/benchmark/comparison_rates_5_to_200_amd.png)

We were not able to run sarvam-30b on Intel Gaudi3 due to software-compatibility issues; we plan to work with the llm-d community to bridge this gap in future.

### Heterogeneous pools — where llm-d wins biggest

The intuition from the table above shows up most starkly when vendors are mixed.

**NVIDIA + AMD (12 pods, granite-4.1-8b).** While k8s throughput plateaus at 10–11 K tok/s, llm-d goes up to 19.4 K — 85% higher throughput. TTFT-wise llm-d is 3.4–5.6× faster at higher rates.

![](prefix-cache-nvidia-amd/benchmark/comparison_mixed_pool_1.png)

**NVIDIA + AMD (12 pods, sarvam-30b).** llm-d brings down TTFT by 2.85–4.54× and increases throughput by close to 3× at rate 200. llm-d wins biggest in this mixed pool — round-robin is most punished by heterogeneous capacity, and llm-d's prefix-aware routing avoids this trap.

![](prefix-cache-nvidia-amd/precise-prefix-sarvam-30b-mixed/benchmark/comparison_rates_5_to_200_mixed.png)

**NVIDIA + AMD + Gaudi (20 pods, granite-4.1-8b).** The 20-pod 3-vendor pool delivers **14.2 K out tok/s peak with llm-d vs 9.6 K with k8s round-robin**. k8s saturates at rate 25 and *declines* to 7.5 K at rate 85 (queue depth dominates) — llm-d delivers **+91% throughput at the same load**. TTFT at rate 85: llm-d 6.8 s, k8s 36.4 s (**5.4× better**).

![](prefix-cache-3vendor/mixed-3vendor-granite/benchmark/comparison_3vendor.png)

## Prefill/Decode Disaggregation

Even within a single vendor, there's a second lever worth pulling. LLM inference has two very different phases: **prefill** processes the input prompt in parallel and is compute-bound, while **decode** generates one token at a time and is memory-bandwidth-bound. When both run on the same GPU instance, a long prefill blocks decode from emitting tokens for everyone else on that pod, spiking TTFT for concurrent users. **P/D disaggregation** routes the two phases onto separate vLLM pools and transfers the KV cache between them over the network, eliminating that interference.

We deployed sarvam-30b on llm-d v0.7 with P/D disaggregation on a single 8-GPU AMD MI325X node using the [well-lit guide](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation). Of the different ways to slice 8 GPUs across the two roles, the configuration that worked best was **4 prefill workers + 4 decode workers, all at TP=1** — eight pods total, with the routing sidecar moving KV cache from prefill to decode over NIXL (NVIDIA Inference Xfer Library). We compared this against a plain Kubernetes baseline of 8 monolithic vLLM pods (TP=1) using round-robin scheduling, as before.

![4P+4D P/D vs 8×TP1 round-robin baseline on 1× MI325X (8 GPUs), sarvam-30b, prefill-heavy](pd-disaggregation/precise-pd-sarvam30b/benchmark/4p4d-amd-prefill-heavy/comparison_4p4d_vs_baseline.png)

With prefill isolated to its own four pods, TTFT stays nearly flat as load increases: at 150 RPS, **TTFT p50 is 4.6 s with PD vs. 17.7 s for the baseline — a 4× improvement**. End-to-end p50 latency is **~2× better** (59 s vs. 113 s), inter-token latency is ~40% lower under load, and the routing sidecar's clean backpressure path delivered **zero failures across 30,000 requests** at the highest rates while the baseline began dropping requests under load.

P/D's peak throughput is ~12% behind the baseline. That's because the baseline keeps all 8 GPUs continuously busy generating output tokens, which maximizes raw throughput; with P/D, only 4 of the 8 GPUs do decode at any instant. But each baseline pod has to interleave prefill and decode on the same GPU, so every new prompt waits behind the in-flight batch — and at saturation that queue grows fast. **For interactive or chat-style workloads where TTFT and end-to-end latency are the SLOs that matter, P/D is the right setting.**

## Future Work

**Cross-accelerator P/D disaggregation.** We plan to take heterogeneous inference to the next level by enabling prefill and decode to run on *different* accelerator types within the same cluster — for example, routing compute-heavy prefill to MI325X nodes and memory-bandwidth-intensive decode to H100 nodes (or vice versa), based on where each phase runs most efficiently. This requires the KV cache transfer library to work across different GPU backends on each end, an active area of development in the llm-d community.

**P/D at larger model and cluster scale.** P/D's gains scale with model size, context length, and deployment size. We plan to repeat the experiment on 120 B+ models with longer contexts and bigger pools, where the prefill-decode interference cost in the monolithic baseline grows — and where the P/D advantage should grow proportionally.

**Sarvam-30b on Intel Gaudi3.** Bringing sarvam-30b's custom vLLM kernels to vllm-gaudi is the missing piece in the 3-vendor sarvam-30b matrix. We are working with the Habana / vLLM-Gaudi communities on closing this gap.

## Takeaways

A few things worth pinning before you go:

- **Heterogeneity is where the routing layer earns its keep.** Single-vendor pools already see +25–85% throughput from llm-d's prefix-cache-aware scheduling; the **+91% / 5.4× TTFT** result on the 3-vendor pool is the real headline because round-robin is most punished by capacity mismatches between vendors. Anyone running a multi-vendor or multi-generation cluster has the most to gain.
- **Workload shape sets the ceiling.** Prefix-cache routing's payoff scales with the prefill-savings opportunity in the workload — long shared system prompts (RAG, chat history, retrieved documents) are squarely in its sweet spot.
- **P/D is a different lever for a different problem.** It doesn't push peak throughput much (it slightly trails the baseline because GPUs split into specialized pools), but it eliminates prefill-decode interference under load — **4× better TTFT at saturation** on the 4P+4D MI325X run, plus zero failures across 30 K requests where the baseline started dropping. For interactive workloads where TTFT is the SLO that matters, that's a step-change.
- **The chart layout scales naturally to N vendors.** One namespace, one InferencePool, one set of selector labels, N `ms-*` Helm releases — adding a fourth or fifth accelerator vendor is a copy of a values file and a one-line edit to the helmfile, with no rewiring of routing or applications.

The net is straightforward: llm-d + vLLM is a single coherent serving layer over heterogeneous accelerators today, open-source, Kubernetes-native, and a CNCF Sandbox project. The remaining open questions — cross-accelerator P/D, larger-scale P/D, sarvam-on-Gaudi — are tractable and being actively worked on in the upstream community.

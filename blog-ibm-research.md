---
title: "Sovereign AI at Scale: Cutting Inference Cost by 30%+ with llm-d on Heterogeneous Clusters"
---

# Sovereign AI at Scale: Cutting Inference Cost by 30%+ with llm-d on Heterogeneous Clusters

*A joint proof-of-concept by IBM Research, Red Hat, and NxtGen Cloud Technologies, India.*

## Why this matters now

As enterprises and governments deploy large language models at scale, **on-premises and sovereign AI is becoming a strategic imperative** — driven by data residency mandates, regulatory frameworks, the need for greater control over core business operations, air-gapped security for sensitive workloads, and the long-term economics of hosting high-volume inference traffic on sovereign and neo-clouds.

Yet the operational reality is hard: enterprises struggle with **underutilized GPUs, rising inference costs, and high latency** from monolithic pipelines, while being locked into single cloud and GPU stacks. Expensive hardware is overused for peak loads while cheaper compute sits idle. Production clusters are rarely homogeneous — they span NVIDIA, AMD, Intel, and other accelerator vendors, and span multiple generations of hardware acquired across procurement cycles. Treating this **heterogeneity as a first-class advantage rather than an operational liability** unlocks real value: lower-cost accelerators absorb prefill-heavy or batch-tolerant workloads, premium hardware is reserved for latency-sensitive paths, stranded capacity gets reclaimed, and the organization avoids single-vendor lock-in at the infrastructure layer.

## The proof point

**IBM Research, Red Hat, and NxtGen Cloud Technologies** came together on NxtGen's sovereign cloud to put this thesis under pressure: can a single Kubernetes-native serving layer take a heterogeneous fleet of NVIDIA, AMD, and Intel accelerators and deliver state-of-the-art performance — without the application ever knowing what it's running on?

The serving layer is **llm-d**, an open-source CNCF-sandbox project that pairs vLLM (high-performance inference engine) with a Kubernetes-native control plane. llm-d delivers three capabilities that together make heterogeneous inference clusters practical at scale:

1. **Prefix-Cache-Aware Scheduling** — routes each request to the GPU most likely to already hold the relevant KV cache in memory, dramatically cutting redundant computation.
2. **Prefill/Decode (P/D) Disaggregation** — separates the compute-heavy prefill phase from the memory-bandwidth-heavy decode phase onto dedicated pools, eliminating resource contention.
3. **Hardware-Agnostic Operation** — presents a single API endpoint regardless of whether the request lands on an NVIDIA H100, AMD MI325X, or Intel Gaudi3, enabling true multi-vendor clusters with no application changes.

We built a 20-GPU heterogeneous pool spanning **4× NVIDIA H100-NVL + 8× AMD MI325X + 8× Intel Gaudi3** on NxtGen's sovereign infrastructure, served `granite-4.1-8b` and `sarvam-30b` (multilingual MoE) on it, and compared llm-d against plain Kubernetes round-robin under a realistic prefill-heavy workload (long shared system prompts plus short questions — typical of RAG, citizen services, and chat).

## Headline results

Across every pool we tested — single-vendor and heterogeneous — llm-d's prefix-cache-aware routing **consistently and significantly beat plain Kubernetes round-robin** on both throughput and time-to-first-token (TTFT):

| Pool | Throughput edge (llm-d vs k8s) | TTFT edge |
| --- | --- | --- |
| 4× NVIDIA H100 | +25–36% | 16× |
| 8× AMD MI325X | +79% | 21× |
| 8× Intel Gaudi3 | +34% | 18× |
| 12 GPUs (NVIDIA + AMD) | +85% | 3.4–5.6× |
| **20 GPUs (NVIDIA + AMD + Gaudi)** | **+91% at peak load** | **5.4×** |

**The win grows with heterogeneity.** Round-robin spreads requests evenly regardless of pod speed, so a single slow vendor becomes a queueing sink that drags total throughput down. llm-d's prefix-cache-aware scheduler routes around saturated pods and concentrates cache hits on warm ones, so heterogeneity is no longer a penalty.

![20-pod 3-vendor pool — llm-d vs k8s round-robin](prefix-cache-3vendor/mixed-3vendor-granite/benchmark/comparison_3vendor.png)

On the 20-pod 3-vendor pool, plain Kubernetes saturates at ~9.6 K output tokens/sec at moderate load and *declines* to 7.5 K under heavier traffic as queue depth dominates. llm-d keeps scaling cleanly to **14.2 K output tokens/sec** — and end-user TTFT at peak load drops from 36 seconds with k8s to under 7 seconds with llm-d.

### Prefill/Decode disaggregation: dramatically smoother latency under load

LLM inference has two very different phases: **prefill** (compute-bound, processes the prompt in parallel) and **decode** (memory-bandwidth-bound, generates tokens one at a time). When both share a GPU, a long prefill blocks decode from emitting tokens and spikes time-to-first-token for everyone else on that pod. **P/D disaggregation** — llm-d's second core capability — routes the two phases onto separate GPU pools so they no longer collide.

We benchmarked this on a single 8-GPU AMD MI325X node serving sarvam-30b on a prefill-heavy workload, comparing 4 prefill + 4 decode replicas (P/D) against 8 monolithic vLLM replicas behind a round-robin service. Throughput per dollar lands in the same ballpark, but **the tail-latency picture changes completely**:

- At moderate load (rate 75 req/s): **TTFT 5× lower** with P/D (0.46 s vs 2.4 s).
- At saturating load (rate 100): **TTFT 8× lower** with P/D (0.59 s vs 4.8 s).
- Inter-token latency is also 30–50% lower under load — streaming feels noticeably smoother.

For interactive applications where TTFT is the SLO that matters, P/D turns "the cluster is melting" into "the cluster is fine" without adding hardware.

## What it means in dollars

Because llm-d squeezes more tokens out of every GPU, you need fewer GPUs to serve the same load.

**Example:** serving `granite-4.1-8b` at **5,000 requests/sec peak** — the kind of traffic profile a national-scale citizen-services chatbot or large enterprise RAG application generates:

- **Plain Kubernetes round-robin:** ~520 GPUs to keep up.
- **llm-d:** ~350 GPUs for the same throughput.
- **Saving: ~170 GPUs ≈ $3.7 M / year** at a representative ~$2.50/GPU-hour blended sovereign rate.

At a fixed accelerator budget, the same advantage shows up the other way: **llm-d serves ~50% more concurrent users with 3–5× faster response time** — directly improving the experience without scaling the cluster.

## Why this matters for the business

This proof point shows that **llm-d can serve as the backbone for AI adoption** by enabling sovereign, population-scale deployment on locally hosted, compliant infrastructure while leveraging enterprise-grade open technologies. Concretely, it lets sectors like **BFSI, government, telecom, and manufacturing** scale GenAI use cases — multilingual citizen services, real-time financial systems, industrial copilots — without prohibitive costs:

- **Data residency, regulatory alignment, and air-gapped security** remain intact, because everything runs on locally hosted infrastructure.
- **Existing GPU investments stop being stranded.** Older or alternative-vendor accelerators get reclaimed into productive serving capacity instead of sitting under-utilized.
- **Single-vendor lock-in evaporates.** The serving layer is hardware-agnostic; the procurement strategy is no longer hostage to one supplier's roadmap or pricing.
- **Cost-per-token drops by 30%+** on realistic enterprise workloads, with the savings compounding across every production deployment in the organization.

## What's next

We are extending the work in two directions:

- **Cross-accelerator P/D** — running prefill on one vendor's GPUs and decode on another's, picking each phase's hardware on its dominant resource. Active development is happening in the llm-d community on cross-vendor KV-cache transfer; we plan to benchmark this once it lands. Larger models (120 B+) on bigger clusters are the natural next target — P/D's tail-latency advantage scales with model size and context length.
- **Continued Intel Gaudi tuning** — Gaudi3 is already integrated and contributing to the heterogeneous pool; we are working with the Habana / vLLM-Gaudi communities on closing the per-pod gap to NVIDIA and AMD on hybrid-architecture (Mamba/SSM) models.

## In short

llm-d makes heterogeneous, sovereign-scale LLM inference **practical, faster, and meaningfully cheaper** — without forcing organizations to standardize on one vendor or rewrite a single line of application code. For enterprises and governments deploying GenAI at population scale, that combination is hard to ignore.

## References

- llm-d project: https://github.com/llm-d/llm-d
- llm-d KV-cache wins blog: https://llm-d.ai/blog/kvcache-wins-you-can-see
- vLLM project: https://github.com/vllm-project/vllm

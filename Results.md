# NVIDIA - 4 GPUs (Prefix-caching)
## Granite-8b  ✅ 
![alt text](<2.1/precise-prefix-granite/benchmark-results/comparison_rates_3_to_22 copy.png>)

**Highlight :  llm-d improves TTFT by upto 16x compared to K8s, and throughput (Output tok/s) by 25-36%**

## Sarvam-30b ✅ 

![alt text](2.1/precise-prefix-sarvam-30b/benchmark/comparison_rates_3_to_50.png)

**Highlight: llm-d delivers 2× the throughput and 22× better TTFT. k8s saturates around rate=25-30; llm-d keeps scaling**

# AMD - 8 GPUs (Prefix-caching)
## Granite-8b ✅ 

![alt text](2.2/precise-prefix-granite-amd-fresh/benchmark/comparison_amd_fresh.png)

**llm-d delivers up to 21× better TTFT and +79% throughput vs plain k8s round-robin on this AMD-only granite deployment**
## Sarvam-30b

![alt text](2.2/precise-prefix-sarvam-30b-amd/benchmark/comparison_rates_5_to_200_amd.png)

**Highlight: While K8s throughput plateaus at 15-17 K tok/s, llm-d goes upto 29K tok/s, 85% higher throughput. TTFT-wise llm-d upto  5x faster for lower rates**

# Gaudi - 8 GPUs (Prefix-caching)

## Granite-8b ✅ 
![alt text](4.2/gaudi-only/benchmark/comparison_gaudi_only.png)
**At saturation (rate 25), llm-d delivers +34% throughput AND ~18× better TTFT vs plain k8s round-robin**
# NVIDIA + AMD - 12 GPUs (Prefix-caching)

## Granite-8b ✅ 

![alt text](4.1/benchmark/comparison_mixed_pool_1.png)

**Highlight: While K8s throughput plateaus at 10-11 K tok/s, llm-d goes upto 19.4K tok/s, 85% higher throughput. TTFT-wise llm-d does 3.4-5.6x faster for higher rates**

## Sarvam-30b 

![alt text](4.1/precise-prefix-sarvam-30b-mixed/benchmark/comparison_rates_5_to_200_mixed.png)

**llm-d brings down TTFT by 2.85-4.54× , increases throughput by close to 3x at rate=200.
llm-d wins biggest in the mixed pool — round-robin is most punished by heterogeneous capacity (slow NVIDIA pods drag k8s peak down to 10K), and llm-d's prefix-aware routing avoids this trap.**

# NVIDIA + AMD + Gaudi - 20 GPUs (Prefix-caching)

## Granite-8b ✅
![alt text](4.2/mixed-3vendor-granite/benchmark/comparison_3vendor.png)

**Highlight: With the optimized Gaudi config (block-size=128, max-num-seqs=256), the 20-pod 3-vendor pool delivers 14.2K out tok/s peak with llm-d vs 9.6K with k8s round-robin. k8s saturates at rate 25 and *declines* to 7.5K at rate 85 (queue depth dominates) — llm-d delivers +91% throughput at the same load. TTFT at rate 85: llm-d 6.8s, k8s 36.4s (5.4× better). 2× improvement over the original un-tuned 4.2 result (7K) purely from the Gaudi tuning; Gaudi is no longer the drag in the heterogeneous pool.**

### Original (un-tuned Gaudi) for reference
![alt text](4.2/benchmark/comparison_3vendor_pool.png)

# NVIDIA - 4 GPUs (PD Disaggregation)
## Sarvam-30b 

![Trying with llm-d 0.7](3.1/precise-pd-sarvam30b/benchmark/comparison_2p2d_highlights.png)

**Highlight: PD reduces tail (inter-token) latency by up to 89%, while closely matching the throughput. PD's ideally works bestfor serving larger models 120b+, hence we do not see throughput gains**

# 4.2 / Gaudi-only Sarvam-30b — DEAD END

**Status: did not produce a usable benchmark.** Captured here so the next person doesn't re-walk the same path.

## What we tried

Run `sarvamai/sarvam-30b` on the same 8×Gaudi3 pool as the granite gaudi-only experiment (`../gaudi-only/`). Same approximate `prefix-cache-scorer` EPP, same flag set, same workload shape. Plus three sarvam-specific tricks:

1. **hostPath HF cache to avoid 8× concurrent downloads.**
   - Pre-downloaded ~60 GiB of weights to `/var/lib/llm-d-hf-cache` on the gaudi node via [`downloader.yaml`](downloader.yaml) (single privileged pod with `huggingface_hub` + `hf download`).
   - SELinux: `chcon -R -t container_file_t /var/lib/llm-d-hf-cache`.
   - Pods mount hostPath at `/root/.cache/huggingface`, set `HF_HOME=/root/.cache/huggingface` and `HF_HUB_OFFLINE=1`.
   - Set `mountModelVolume: false` in chart values to skip its init download.
   - **This part worked.** Download took 498 s (~8 min) once. Subsequent pod loads read from local NVMe (~45 s/shard cold; faster on second pod due to OS page cache).

2. **transformers ≥ 5.0 upgrade at container start.**
   - v0.7.0 hpu image ships `transformers 4.57.6`, but sarvam-30b's MoE class needs ≥ 5.0. `pip install --upgrade 'transformers>=5.0.0'` in the entrypoint.
   - vllm + vllm-gaudi pin `<5` so pip prints a warning, but installs anyway and runtime imports work.
   - **This part worked.**

3. **Sarvam vllm hotpatch (drop-in `sarvam.py` + registry edit).**
   - Without this, vllm falls back to Transformers backend, which crashes on shape mismatch in `modeling_sarvam_moe.py:418`: `bsz, q_len, _ = hidden_states.size()` (got 4D, expected 3D).
   - The sarvam team publishes [hotpatch_vllm.py](https://huggingface.co/sarvamai/sarvam-30b/raw/main/hotpatch_vllm.py) which:
     - downloads `sarvam.py` from HF and writes it to `vllm/model_executor/models/sarvam.py`
     - edits `vllm/model_executor/models/registry.py` to add `SarvamMoEForCausalLM`
   - We inlined the patch into the entrypoint (idempotent on restart).
   - **This part loaded the model**, vllm logged `Resolved architecture: SarvamMoEForCausalLM`, and engine init completed.

## Where it broke

`POST /v1/completions` returned **HTTP 200** but with **degenerate output**:

```
prompt:   "Hello, my name is"
output:   " is is is is is is is is is is is is is is is is is is is is"
```

Both first (cold compile) and second (warm) requests took ~65 s for 20 tokens. That's a forward-pass crawl AND a degenerate token distribution.

### Diagnosis

The upstream `sarvam.py` was written against vllm `0.15.x` core APIs (`hotpatch_vllm.py` literally pins `vllm==0.15.0` in its installer). vllm-gaudi `0.16.0` has core API drift — module imports succeed (no ImportError) but the model's forward pass produces wrong logits, indicating one of:
- KV cache layout / paged-attention API mismatch
- MoE expert routing call shape
- RMSNorm / RoPE arg order changed
- HPU-specific kernel binding silently no-ops

Without source-level debugging across `sarvam.py` × `vllm/model_executor/layers/*` × `vllm_gaudi/v1/worker/hpu_model_runner.py`, we can't identify which layer is producing wrong activations. That's a multi-day project.

## Why we stopped

- `--enforce-eager` would test if the bug is in HPU graph capture vs the model itself — worth ~10 min.
- Downgrading vllm to 0.15 would break vllm-gaudi (which is 0.16).
- Patching sarvam.py to match vllm 0.16 API is real engineering work, not flag tuning.

The user's ask was a quick comparison; this isn't quick. **Documenting and moving on.**

## What's preserved here

- [downloader.yaml](downloader.yaml) — proven hostPath HF cache primer (works for any HF model, not just sarvam).
- [smoke-warmup.yaml](smoke-warmup.yaml) — single-pod smoke with the full hotpatch sequence inlined.
- [ms-kv-events/values.yaml](ms-kv-events/values.yaml) — full ms-chart values with HF cache mount + transformers upgrade + sarvam.py hotpatch + `VLLM_SKIP_WARMUP=true`.
- [helmfile.yaml.gotmpl](helmfile.yaml.gotmpl), [deploy.sh](deploy.sh), [httproute.yaml](httproute.yaml) — same shape as `../gaudi-only/`, namespace `llm-d-sarvam-gaudi`, release postfix `sarvam-gaudi`.
- The hostPath cache at `/var/lib/llm-d-hf-cache/hub/models--sarvamai--sarvam-30b/` is intact on the gaudi node — if we revisit this, we don't re-download.

## When to revisit

Bring sarvam-30b back to gaudi when **any** of:
- vllm-gaudi releases ≥ `0.7.x` aligned with vllm 0.17+ AND sarvam team publishes a corresponding `sarvam.py`.
- The sarvam team validates their model on Gaudi explicitly (announces in HF model card).
- We have time to bisect `sarvam.py` against vllm 0.16 internals (probably 1–2 days of focused debugging).
- A pre-baked `llm-d-hpu` image ships sarvam support natively (no hotpatch needed).

# Escalation: GQA Decode SDPA Performance Gap on gfx942/gfx950

**To:** CK SDPA / hipDNN engine team  
**From:** ysoliman  
**Date:** 2026-06  
**Branch:** users/ysoliman/heuristic-plugin-prototype

---

## Bottom Line

GQA decode (sequence length = 1) SDPA performance on gfx942 (MI300X) is **0.09-0.17
TFLOPS** - approximately 400-600x slower than MHA prefill on the same hardware. This is a
kernel gap, not a selection gap. The hipDNN heuristic prototype can identify and classify
these shapes; there is no performant kernel candidate to route them to yet.

This is the dominant performance liability for LLM inference workloads on AMD hardware.
Every autoregressive generation step (token-by-token decode) hits this regime.

---

## Benchmark Data (gfx942 / MI300X, ROCm 7.2)

| Regime       | Condition       | TFLOPS    | vs. MHA prefill |
|--------------|-----------------|-----------|-----------------|
| GQA_DECODE   | Sq = 1          | 0.09-0.17 | ~400-600x slower |
| GQA_PREFILL  | Hq > Hkv        | 3-6       | ~10-18x slower  |
| D256_PREFILL | D = 256         | ~29       | ~2x slower      |
| MHA_PREFILL  | else            | 55-70     | baseline (OK)   |

Full benchmark data: `gfx942_sdpa.csv` (23 shapes, bf16).  
gfx950 benchmark data is blocked by the current PyTorch environment reporting
`torch.cuda.is_available() is false` on gfx950, but the SDPA verifier did confirm the same
hipDNN FlatBuffers/heuristic path on gfx950.

---

## Root Cause

These are kernel gaps, not selection gaps. The available engines do not provide a performant
implementation for:

- single-token decode (`Sq=1`) with grouped-query attention (`Hq > Hkv`)
- high head dimension (`D=256`) tile configurations

The heuristic routing infrastructure is in place, but routing cannot improve a shape until a
better candidate engine exists in the candidate list.

---

## Evidence: Heuristic Classification Works

The SDPA regime classifier confirms these shapes are being identified from the serialized
hipDNN graph. The classifier fired through standalone verifier paths on both gfx942 and
gfx950:

```text
[SDPA_HEURISTIC] PolicySetSerializedGraph size=664
[SDPA_HEURISTIC] regime=MHA_PREFILL Hq=8 Sq=16 D=64 - OK (55-70 TFLOPS)
```

The same classifier contains explicit logs for the critical regimes:

```text
[SDPA_HEURISTIC] regime=GQA_DECODE Hq=<N> Hkv=<M> D=<D> - CRITICAL gap (0.09-0.17 TFLOPS)
[SDPA_HEURISTIC] regime=GQA_PREFILL Hq=<N> Hkv=<M> Sq=<S> - kernel gap (3-6 TFLOPS)
```

The routing integration plan is already written in `heuristic_plugins/Q3_ROUTING_SPEC.md`.
When a performant CK SDPA engine is available and its engine ID is known, the heuristic can
activate routing with a small follow-up change.

---

## Ask

1. Prioritize GQA decode in CK SDPA engine work. The `Sq=1` decode case is the highest-impact
   gap for autoregressive LLM inference.
2. Target the `D=256` tile mismatch as a secondary gap affecting larger head dimensions.
3. Publish or confirm the CK SDPA engine ID when the engine lands so the heuristic can route
   `GQA_DECODE` and `GQA_PREFILL` to it.

---

## References

- Benchmark data: `gfx942_sdpa.csv`
- Heuristic plugin: `heuristic_plugins/sdpa_heuristic/sdpa_heuristic.cpp`
- Routing spec: `heuristic_plugins/Q3_ROUTING_SPEC.md`
- Build log: `~/heuristic_build_log.md`

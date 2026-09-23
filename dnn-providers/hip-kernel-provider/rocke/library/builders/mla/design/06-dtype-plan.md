[← MLA design doc index](../DESIGN.md)

## 6. Dtype plan

| Phase | Dtype | Arch |
|---|---|---|
| 1 | bf16 | gfx942 + gfx950 (prefill and decode-absorb) |
| 2 | fp8 e4m3 (KV cache) | gfx950+ only (fp8 prefill and fp8 decode-absorb) |

**fp8 approach:** Follow the existing sync-dequant pattern from
`library/kernels/common/fmha_fwd_fp8.py` and §7.3 of the gfx950 `ALGORITHM.md`
(that document's numbering, not this one's):
store `c_KV` and `K_rope` as fp8 e4m3 with per-block scale factors; dequant to
bf16 in LDS before the MFMA. The W_UK and W_abs weights remain bf16: quantizing them
adds accuracy risk, and the KV cache is what fp8 is being applied to here. Note this is
a scoping decision, not a claim that weight traffic is negligible — §4 records `W_abs`
at ~192 MiB per decode step, which dominates at low batch.

**fp8 is excluded from gfx942.** The gfx942 decoder does not have the
`ds_read_tr` transposition facility that makes fp8 dequant efficient on gfx950.

---


# dsv41_mono (gfx950)

These are rocKE stage functions for ATOM's DeepSeek-V4.1 FlyDSL mono decode kernels. They use the hybrid path: each
is a rocKE `KernelDef` lowered as an LLVM device function, which ATOM links in place of one FlyDSL stage body. The
knob is `ATOM_DSV41_RK_EXTERN`, in the ATOM-dsv41 tree's `atom/models/deepseek_v41/mono/kernels/rk_extern.py`; it
is off by default, and off the kernels are byte-identical to ATOM's. This family reuses
`qwen38_mono/device_function.py` and does not modify it.

| module | what |
|---|---|
| `stage_wqkv.py` | K1 `wqkv_a`. It re-implements ATOM's `gemv_fp8_loads` + `load_x8` + `stage_wqkv` bit for bit: an FP8 E4M3 weight with E8M0 32x32 block scales, times the tokens' MXFP8 rows from the `x8` / `x8s` mailboxes, then `v_mfma_scale_f32_16x16x128_f8f6f4`, a wave-order sum, bf16 RNE, and `(f32 bits, tag)` puts. Symbol `rocke_dsv41_wqkv_n<rows>` (rows 1-16, one token tile). Knobs `WqkvKnobs`: `poll`, `pause`, `poll_max`, `w_cpol`, `x_first`, `red_bar`, `sop` (op_sel-packed scales), `ra` (LDS read-ahead). Test mutants: `put_bit`, `chain_drop`, `row`, `scale_byte`, `wave_order` (report-only), `opsel`. CLI: `python -m kernels.gfx950.dsv41_mono.stage_wqkv --out DIR [--rows 1,..,16] [--knobs k=v,..] [--atom-root ATOM]` writes `wqkv_n<rows>.{ll,json}`; then run ATOM with `ATOM_DSV41_RK_EXTERN=wqkv=DIR`. |
| `stage_wqb.py` | W13 WP4 (4.2) K1 `wq_b` (+ the head's GPT-J RoPE, bf16 `q_out`) on `stage_wqkv`'s core, rows 1-16, TP2. Symbol `rocke_dsv41_wqb_n<rows>`; `ATOM_DSV41_RK_EXTERN=wqb=DIR`. |
| `stage_woa.py` | W13 WP4 (4.2) K2a grouped `wo_a` task + ATOM's `quantize_fp8` into `x8b` / `x8bs`. Symbol `rocke_dsv41_woa_n<rows>`; `ATOM_DSV41_RK_EXTERN=woa=DIR`. |
| `stage_wob.py` | W13 WP4 (4.2) K2a `wo_b` block (two tasks a CTA, bf16 pair pushes to every rank's `attn` region). Symbol `rocke_dsv41_wob_n<rows>`; `ATOM_DSV41_RK_EXTERN=wob=DIR`. |
| `stage_shared.py` | W13 WP4 (4.2) K2b shared expert gate / up GEMV + `_silu_mul` + `smid_quant`. Symbol `rocke_dsv41_shared_n<rows>`; `ATOM_DSV41_RK_EXTERN=shared=DIR`. |
| `stage_moe.py` | W15 PORT K2-moe: the K2b routed-MoE stage bodies, bit for bit: `stage_xq` (`xq_c<x_cm>`), `stage_route` (`route_n<S>e<E>k<K>`), `stage_down` for one / two tasks (`down_n<S>e<E>k<K>g<G>[p]t<1|2>`, `p` token parallel), `stage_reduce` (`reduce_n<rows>`), the shared unit past one token tile (`shm_t`), and `stage_ug`'s function under `ug_n<S>g<G>` for every width (multi-tile: a pick tile's 16 columns). Every TP2 served width (1-48 rows, target 384 / top-6 and draft 128 / top-3). ATOM side: `ATOM_DSV41_RK_EXTERN=x8q=D;route=D;down=D;reduce=D;ugx=D;shx=D;shm=D;ug=D;shared=D` (`rk_extern_moe.py`). Knobs `MoeKnobs` (`nan`, `pause`, `poll_max`, `w_cpol`, `sop`, `r_max`, `sh_first`, `wfirst`, `pslot`, `smap`, `chains`), mutants per stage (`MUTANTS`), `standalone()` builds for the COMGR-vs-Rust-direct rig. CLI: `python -m kernels.gfx950.dsv41_mono.stage_moe --out DIR [--stages ..] [--rows ..] [--groups 1,2] [--knobs k=v,..] [--atom-root ATOM]`. Results: `experiments/dsv41-mono/w15/PORT-K2-moe.md`. |
| `wide_gemm.py` | W13 WP6 wide path (standalone kernels, not stage functions): the FP8 32x32 dense GEMM of ATOM's native group32 linears at 16-384 rows (`rk41w_*`, ATOM's preshuffled weights read directly, MXFP8 activation quantization fused or as its own launch `rk41q_*`), and ATOM's `quantize_fp8` bit for bit (`rk41qa_*`). Loaded by `ATOM_DSV41_RK_WIDE` / `ATOM_DSV41_RK_SEAM` (`atom/models/deepseek_v41/rk/wide.py`). |
| `router_topk.py` | W13 WP6 routing fold: router GEMM + sqrtsoftplus top-6 in one launch (`rk41r_*`), `ATOM_DSV41_RK_FOLD`. |
| `moe_floor.py` | W12 P4: the routed-MoE bytes floor prototype. |

The four WP4 GEMV stages reuse `stage_wqkv`'s core, `WqkvKnobs` (`sop` / `ra`) and its mutants, with the same CLI
(`--out DIR --rows .. --knobs ..`); their ABIs are in each module's docstring. Several stages combine with `;`:
`ATOM_DSV41_RK_EXTERN=wqkv=D;wqb=D;woa=D;wob=D;shared=D`. Results (bitwise in-place gates, timing ratios):
`copy-rocm-lib/experiments/dsv41-mono/wps/WP4.md`.

Results of the WP6 kernels: `copy-rocm-lib/experiments/dsv41-mono/wps/WP6.md` (all default off: they do not beat ATOM's
kernels end to end).

The ABI of `wqkv`:

```
void (i32 tag, i32 task, i64 w, i64 ws, i64 x8, i64 x8s, i64 qkv, i32 xl, i32 xsl, i32 red, i32 lds_base)
```

- `x8` / `x8s` / `qkv` are the mailbox region addresses (scratch + offset).
- `xl` / `xsl` / `red` are LDS word offsets from `lds_base`, which ATOM passes as the byte address of its `red` buffer.

Gates, timing and the reverse-engineering notes are in `copy-rocm-lib/experiments/dsv41-mono/reverse/MONO-STAGES.md`.
The harness is in `hackathon-prep/dsv41/p2/`: `p2_rig.py` (in-situ bitwise A/B and timing), `p2_dump.py`
(stage-marked dumps) and `p2_run.sh`.

## cpp-parity exemption

Device functions and `tile.tagged_poll` / `mfma_scale_mx` are Python-engine-only. This is the same exemption as
`qwen38_mono` (its README, "cpp-parity exemption"). Nothing here is under `rocke.helpers`.

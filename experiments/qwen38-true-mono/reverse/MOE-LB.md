# Large-batch MoE, reverse-engineered: rocKE (P-L4 `moe_lb`, WP-L2d `moe_lb2`) vs FlyTape's tiled tape

For the MoE rebuild (WP-L2d). The comparison is static ISA, phase by phase, made with the reverse toolkit
(`hackathon-prep/qwen38/truemono/reverse/`, see `FLYDSL-STAGES.md` section 0). Counts are instruction counts in a
phase's code, not time. Ratios only.

## Data

| side | build | job |
|---|---|---|
| FlyTape | tiled tape, `bench_qwen4_full.py --tp 1 --precision mxfp4`, B16 / B64 x P128. The whole step is one persistent kernel (`launch`); the MoE phases are `moe_router`, `moe_up`, `moe_down`, `moe_combine`. TP1's 640 intermediate rows keep the two expert phases (no `moe_expert`). | `rv-ft1` (68240683, q38 container, 1 GPU) |
| rocKE P-L4 | `moe_lb.MoeKnobs(tp=1, act="fp4", bucket=64 / 16, sop=True)` (P-L4's best arm), plus fp8 b64 and tp2 fp4 b128 | `rv-dump4` (68240929) |
| rocKE WP-L2d | `moe_lb2.Moe2Knobs(tp=1, act="fp4", bucket=64)` (+ `tpp=2`, b16, fp8), the tree at `moe_lb2.py` sha256 `580f8526...` | `rv-lb2a` (68240820) |

- **FlyTape split.** The phases come from phase markers (`rv_ft.py` wraps every `flytape.spec.Phase` body; FlyTape is not
  edited). The marked tapes have the plain tapes' `.text` (2 of 2).
- **rocKE split.** `ROCKE_DEBUG_LOC=1` line tables give each instruction's Python function (`rv_isa.py --pyfunc`). The
  debug builds have the plain `.text` (14 of 14).
- **Phase mapping.**

  | rocKE functions | phase |
  |---|---|
  | `route` | route |
  | `sort` / `deal` / `lds_put` | sort |
  | `gu_unit` | gate_up (= FlyTape `moe_up`) |
  | `dfirst` / `dn_unit` | down (= `moe_down`) |
  | `combine_phase` | combine (= `moe_combine`) |
  | `gbar` / `_pollge` / `_aadd` / deadline / err | sync |

- **FlyTape copies.** FlyTape traces 6 copies of each MoE phase (one per traced layer position). Copy 0 is compared.
  Router / up / down copies differ by under 2 % in instructions; the combine copies by up to 1.8x (copy 0 is the
  shortest).
- **Reports:** `hackathon-prep/qwen38/truemono/reverse/results/lb_b64.{md,json}` and `lb_b16.*`. The `lb2` function-level
  splits are in `hackathon-prep/logs/q38/rv-lb2a/lb2/*.dbg.fn.split.json`.

## 1. Phase by phase (B64; B16 is the same code shape)

Table values are rocKE `moe_lb` sop / FlyTape.

| phase | instructions | MFMA | weight loads | activation / scale loads | in flight (max) | serial MFMA chains / LDS-serial / scale rewrites | other |
|---|---|---|---|---|---|---|---|
| route | 1396 / 2786 (FlyTape includes the router GEMV: 16 bf16 MFMAs) | 0 / 16 | - | - | 1 / 37 | - | rocKE: 426 `s_nop` cycles, all after VALU (DPP argmax chains of the top-10); FlyTape: 643 after VALU, 2 hard-spin polls |
| sort | 293 / (inside `moe_up` / `moe_down`: `_build_lists`) | - | - | - | - | - | both rebuild the per-expert token lists in every CTA |
| gate_up | 719 / 1608 | 80 / 160 | `dwordx4 nt` 80 / 160 | x: 20 plain `dwordx4` after one acquire / 40 `dwordx4 sc1`; scales: 20 `dword nt` + 5 / 40 + 40 | 25 / 22 | 0 / 0 / 0 vs 1 / 0 / 1 | same per-MFMA ratios: 1 weight `dwordx4` per MFMA, 1 x `dwordx4` and 1 weight-scale dword per 4 MFMAs. FlyTape's unit is twice as long (UP_GROUP = 2 tiles). |
| down | 337 / 712 | 20 / 40 | `dwordx4 nt` 20 / 40 | h: 5 `dwordx4 sc1` + 5 `dword sc1` / 10 + 10; **weight scales 20 `dword nt` / 12 plain `dword`** | 40 / 14 | 0 / 0 / 0 vs 0 / 0 / 1 | FlyTape: DOWN_TILES = 4 output tiles share one h fragment (4 back-to-back MFMAs into 4 accumulators). rocKE is the same (`dt=4`), but loads one scale dword per MFMA. |
| combine | 315 / 143 | - | - | rocKE 44 `dwordx4 sc1` / FlyTape 11 `dwordx2 sc1` + 11 `dword sc1` | 12 / 0 | - | FlyTape: narrow loads, slot sums in order |
| sync | 143 executed instructions in 7 helpers (plus 270 never-executed padding `s_nop` after the kernel's end) / grid barriers between phases (inside each phase's code: 10 CTA barriers) | - | - | - | - | - | rocKE flow sync: 6 full `vmcnt(0)` drains for 2 loads, a drain before a barrier, 44 `s_nop` cycles (asm hazard pads `hz`) |

Whole kernels:

- **rocKE `moe_lb` sop b64:** 234 VGPRs, 106 SGPRs, **95 SGPR spills** (b16 114, tp2 b128 164).
- **FlyTape tape:** 409 VGPRs (incl. 153 AGPRs), 106 SGPRs, 1725 SGPR spills over the whole step. Its MoE phases hold 7-10
  `v_writelane` each; the spills sit mostly in the attention / GDN phases.

**Reading.** P-L4's per-unit loops are already at FlyTape's structure:

- no serial MFMA chains;
- no MFMA waiting on LDS;
- no scale rewrites (`sop`);
- `nt` weights;
- deeper in flight in the down phase.

Three code-level differences are left:

1. **Down weight scales.** rocKE loads one scale dword per MFMA (20 `buffer_load_dword nt` for 20 MFMAs); FlyTape loads
   about 0.3 per MFMA (12 for 40). The `dsop` knob of `moe_lb` (scales in the 4-chunk layout, one load per 4 chunks)
   is off in P-L4's best arm.
2. **Route.** The top-10 runs as dependent DPP argmax chains: 426 of the phase's `s_nop` cycles follow a VALU,
   e.g. `v_max_u32_dpp` -> `v_max_u32_dpp`.
3. **Sync.** The flow counters add full drains and hazard pads (44 nop cycles) in code that FlyTape replaces with one
   grid barrier per phase.

Everything else in FlyTape's MoE is longer per unit (2x the MFMAs per gate_up unit) or narrower (combine), not leaner.
P-L4 measured rocKE behind FlyTape's experts + combine at r32 and above (1.03-1.35, `large-batch/P-L4-tables.md`). The
ISA does not show a per-unit code deficit large enough to explain that, so the remaining gap is in:

- the deal / line;
- the hand-off structure (flow vs grid phases, `hand` / `combine` knobs);
- the per-wave work size (`gcb`, `tpp`).

## 2. `moe_lb2` (the rebuild, sha `580f8526`): three code regressions against both P-L4 and FlyTape

Function-level split of `lb2moe_tp1_fp4_b64` (12323 instructions; 256 VGPRs, **298 SGPR spills**; `tpp=2`: 16845
instructions, 318 SGPR spills):

1. **`down_stream`** (4590 instructions, 700 MFMAs, 700 `dwordx4 nt` + 15 `dword nt` loads).
   - **Serial accumulator chain:** 618 of 700 MFMAs accumulate into the previous MFMA's destination.
   - **Every MFMA result is stored at once:** 300 `buffer_store_dwordx4 ... sc1`, each behind an `s_nop 6` hazard pad.
     That is 2087 nop cycles in the stream (4302 with `tpp=2`).
   - **Load addresses come from spilled SGPRs:** each load's address is computed by `v_readlane` of a spilled SGPR,
     then `s_mul`.

   ```
   v_mfma_scale ... v[56:59], v[64:67], v[48:51], v[56:59], v16, v42
   s_lshl_b32 s38, s25, 6
   s_nop 6
   buffer_store_dwordx4 v[56:59], v47, s[72:75], s38 offen sc1
   ...
   v_readlane_b32 s25, v252, 28
   s_mul_i32 s34, s25, 0x1400
   buffer_load_dwordx4 v[60:63], v43, s[4:7], s34 offen nt
   ```

   For comparison:
   - P-L4's `dn_unit`: 0 chained MFMAs and 4 stores for 20 MFMAs.
   - FlyTape's `moe_down`: 4 independent accumulators per h fragment, and immediate `offset:1024/2048/3072` on one VGPR
     base.
2. **`stage_x`** (2198 instructions, 80 LDS DMAs).
   - Every `buffer_load_dwordx4 ... lds` is preceded by 4 `v_readlane` that restore its buffer resource from the spill
     VGPR, an `m0` write and an `s_nop 4`: 721 `v_readlane` and 400 nop cycles in the phase.

   ```
   v_readlane_b32 s14, v251, 31; s_mov_b32 m0, s14
   v_readlane_b32 s16..s19, v251, 47..50
   s_nop 4
   buffer_load_dwordx4 v46, s[16:19], 0 offen lds
   ```

3. **SGPR spills** come from `_rsrc` (158 `v_writelane` at entry: buffer resources built once and held) and `combine`
   (115).

**Knob candidates for the rebuild:**

- **Down accumulators:** keep a down tile's partial in registers across its K chunks, store once per tile (P-L4 / FlyTape
  form), and give the `dt` tiles their own accumulators sharing one h fragment.
- **Down weight scales in the 4-chunk layout** (`dsop`), one load per 4 chunks.
- **Resources and offsets:** build buffer resources lazily at the use site (P9: 23 SGPR spills to 1). Fold per-load
  offsets into buffer immediates on one base, instead of `s_mul` from a restored SGPR.
- **DMA staging:** hoist the DMA resource and the `m0` base out of the staging loop and step `m0` with `s_add`. Compare
  against `xst=vgpr` (loads + `ds_write`).
- **Pads:** keep `hz` only where a VALU write of an SGPR precedes asm VMEM. The `s_nop 6` here is the compiler's MFMA ->
  store hazard: it goes away when stores are batched.
- **Route:** interleave the DPP chains of several rows / keys to fill the `s_nop` slots.

## 3. How to re-run

```bash
cd hackathon-prep/qwen38/truemono/reverse
bash sync_rv.sh <sfx> overlay        # stage the tools + copy-rocm-lib's current large_batch/*.py
bash ../P6/submit_cpu.sh <log> --tag rv-lb2<x> --gpus 0 --cpus 32 --mem 128 --time 00:40:00 --mode container \
  --stage rv-h-<sfx> --stage rocke-qk-cs7 --env RV_H=rv-h-<sfx> --env RV_RK=rocke-qk-cs7 \
  --env "RV_LB2=lb2moe_tp1_fp4_b64 lbmoe_tp1_fp4_b64_sop" -- 'bash /job/stages/rv-h-<sfx>/reverse/rv_lb2_job.sh'
# function-level split against the staged sources (the tree may have moved on since):
MSYS_NO_PATHCONV=1 python rv_isa.py split X.dbg.hsaco --out X.split.json --pyfunc "large_batch/moe_lb2?\.py" \
  --pyfunc-skip run_layer_experts,stream_layer,stream_segment --remap "/job/rk2=<staging copy>/overlay" \
  --llvm-bin <llvm with llvm-symbolizer>
python rv_lb.py --rocke LB.split.json --ft FT.split.json --md out.md      # FlyTape: rv_ft_job.sh
```

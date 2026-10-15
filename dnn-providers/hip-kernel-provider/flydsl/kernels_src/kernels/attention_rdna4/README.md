<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL attention for RDNA4 (gfx12)

Vendored kernel sources for the `hipkernel:flydsl_sdpa_aiw` engine: AOTriton's FlyDSL
forward kernel for gfx12, an engine of its own beside `hipkernel:flydsl_sdpa_rdna` (one
kernel family per engine). The RDNA3/RDNA4 kernels under `../attention_rdna/` stay that
engine's; this one is faster on medium and long sequences, and which of the two runs a
graph both serve is autotune's (UHD's) to decide.

| File | Upstream | Role |
|---|---|---|
| `flash_attn_func_gfx1201_aiw.py` | AOTriton `modules/flash/flyc/flash_attn_func_gfx1201_aiw.py` | the forward kernel and its launcher |
| `fmha_tuning_gfx1201.py` | same directory | per-head_dim schedule (block sizes, waves, prefetch, shards) |
| `fmha_common_gfx1201.py` | same directory | tile staging, masking and addressing helpers |
| `fmha_abi_gfx1201.py` | same directory | host-side argument packing (varlen, window, dropout, LSE) |
| `flyc_polyfill.py` | same directory | helpers absent from the pinned FlyDSL kernel tree |
| `philox.py` | same directory | the dropout RNG (imported; dropout is not served) |

Upstream: https://github.com/ROCm/aotriton at
`b5e8cfb909e554c62081f56055b11d6ba3fd6bfa`. AOTriton pins FlyDSL 0.3.1 for these
files; they build on the 0.3.4 compiler and kernel tree this provider pins
(`generators/_flydsl_env.py`) with the modifications below, and a build of them matches
the timings of the kernel AOTriton ships to PyTorch. The files carry their own
Apache-2.0 headers (FlyDSL Project Contributors); AOTriton's repository is MIT.

## Modifications

0. **Package imports.** Upstream's directory is flat and its files import one another as
   siblings (`import fmha_common_gfx1201 as fmha`). Here they are a package, so those
   imports become `from kernels.attention_rdna4 import fmha_common_gfx1201 as fmha`
   (code only; usage examples in docstrings are left as upstream wrote them).
1. **`_if_then`.** `fmha_common_gfx1201.py` enters an `scf.if` then-block through
   `kernels_common._if_then(if_op)`, a context manager in FlyDSL 0.3.1's kernel library
   that the 0.3.4 tree no longer has. It is defined in `flyc_polyfill.py` instead, and
   the five call sites use it from there.
2. **The LSE of a row no key reaches is -inf.** Upstream stores +inf there, the
   convention AOTriton's backward pass needs (it subtracts the LSE, so +inf zeroes the
   row); hipDNN, cuDNN and their references use -inf.

3. **Attention sinks** (`flash_attn_func_gfx1201_aiw.py`, `fmha_tuning_gfx1201.py`). A
   `sink` build (input metadata, a baked axis) takes trailing `SINK, sink_stride`: one
   logit per query head, in Q's dtype, that joins every row's softmax as a key with no
   value, as gpt-oss defines it and as the RDNA kernel's modification 21 does. Before the
   logsumexp, a row's max becomes `max(m, sink * log2e)`, its sum gains the sink's
   term, and O is rescaled to match; a row no key reaches gets O = 0 and LSE = the sink.
   Builds without `sink` keep the two arguments and never read them. Checked on gfx1201
   against an f32 reference: O at bf16 precision and the LSE within 2e-6, causal and not,
   GQA and MHA; time unchanged on long sequences.
4. **An additive mask (attn_mask) with a causal mask or a window**
   (`flash_attn_func_gfx1201_aiw.py`). Upstream refuses the pair, following PyTorch,
   which leaves it undefined. hipDNN and cuDNN define it: the mask's values are added,
   and the causal mask or window still removes the keys it removes. The kernel already
   applies them in that order in every tile (scale, additive mask, then the causal or
   window mask writing -inf), so the change only removes the refusal. Checked on gfx1201
   against an f32 reference, with and without a sink: O at bf16 precision and the LSE
   within 2e-6.
5. **An f32 additive mask, and a key axis with any stride** (`flash_attn_func_gfx1201_aiw.py`,
   `fmha_tuning_gfx1201.py`). Upstream reads the mask in Q's dtype with a contiguous key
   axis, eight adjacent keys per v8 load. Two build axes widen it:
   - `bias_f32` reads an f32 mask (a v8 f32 load on full tiles);
   - `bias_key_strided` takes the per-element path on every tile, with the key column
     scaled by a new trailing argument, `stride_b_seq_k`. Stride 0 broadcasts one
     value along a row, and a key-major mask works too.

   The batch, head and query strides were already runtime arguments, so any of
   them may be 0. Every build carries `stride_b_seq_k`; one without
   `bias_key_strided` never reads it. The host now also checks the mask's dtype
   (upstream's forward did not, so an f32 mask passed to a build reading Q's dtype
   was silently misread) and refuses a non-unit key stride on a build without the
   axis. Checked on gfx1201 against an f32 reference: O at bf16 precision and the
   LSE within 2e-6, across f32 and bf16 masks, broadcasts along every axis, and a
   key-major layout. Cost: `bias_key_strided` is 17-20% slower than the v8 path on
   a contiguous mask, so it is for masks that need it. A full-size f32 mask reads
   twice the bytes of a bf16 one, and on long sequences that read sets the pace.
6. **V's head count on its own** (`flash_attn_func_gfx1201_aiw.py`, `fmha_abi_gfx1201.py`).
   Upstream groups V's heads with K's. hipDNN only asks that each divide Q's head
   count, so a new trailing `num_head_v` argument groups V separately. That is
   one uniform scalar divide in every build, and it equals K's grouping when the
   counts match. `prep_tensors` drops its K-equals-V check for the forward call only
   (`v_heads_own`). Checked on gfx1201 with V heads above, below and equal to K's,
   including one V head under 16 query heads.
7. **An LSE with strides of its own** (`flash_attn_func_gfx1201_aiw.py`). Upstream
   writes a compact LSE, (B·H, Sq) or Transformer Engine's (B·Sq, H), choosing between
   them from the varlen layout bits. hipDNN's stats tensor is (B, H, Sq, 1) in any
   layout. Three trailing arguments, `lse_stride_b/h/s`, carry its strides. A
   nonzero query stride selects them at the one store per row, so every build
   takes both forms. The host accepts a rank-3 or rank-4 LSE on dense calls and
   passes zeros for the compact one. Checked on gfx1201 with (B, H, S), (B, S, H) and a
   view carved from a padded buffer: every element written and within 2e-6 of the
   reference; time unchanged.
8. **A sink of -inf on a row no key reaches** (`flash_attn_func_gfx1201_aiw.py`). The
   sink joins the row's softmax against `m_top`, the larger of the row's max and the
   sink; a row with no keys took the sink's logit alone, so a -inf sink made `m_top`
   -inf and the sink's term exp2(-inf - -inf) NaN (poison under fast-math), giving a NaN
   LSE where hipDNN expects -inf. `m_top` there is floored at the kernel's own finite
   m floor (-3.40282e38), so the term is exp2(-inf) = 0, `l` is exactly 0 and the LSE
   takes modification 2's -inf. A row that reaches keys, or a finite sink, is unchanged.

Not modified, but checked: **MLA's head_dim 192 with a value head_dim of 128** builds
and runs as upstream wrote it. On gfx1201 against an f32 reference, causal and not and
under GQA, O is at bf16 precision and the LSE within 2e-6.

## Preconditions (not modifications)

- **Head dims.** A build's head_dim is rounded up to a tile on the ladder
  `16 ... 256, 384, 512`. Only `fmha_tuning_gfx1201.plan()` sets `padded_head`
  when the two differ, so builds must go through it; the keyword front end does not.
  A padded build serves any real head dim up to its tile, passed at runtime as
  `hdim_qk`/`hdim_vo`, **if it is a multiple of 8**. Others read V misaligned and
  return a wrong O while the LSE stays right; PyTorch pads to a multiple of 8 before
  calling, so upstream never sees one. That is the rule the RDNA kernels' generic
  objects already have, so it costs no coverage. Checked on gfx1201 for real dims
  72 and 80 on the 80 tile and 136 on the 160 tile, and for 256, 384 (where PyTorch
  ships no kernel and this one is 1.6x faster) and 512.
- **Windows.** Causal builds take any window at runtime: key j is kept iff
  `i - window_left <= j <= i + window_right`. hipDNN's bottom-right alignment is
  the same window shifted by `Skv - Sq`. Checked on gfx1201 for sliding windows,
  two-sided bands and the diagonal, top-left and bottom-right, Sq above and below
  Skv. A non-causal build refuses a window.

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).

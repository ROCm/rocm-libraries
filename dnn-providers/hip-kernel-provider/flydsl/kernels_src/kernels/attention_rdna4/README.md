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
| `philox.py` | same directory | the dropout RNG (the `has_dropout` objects, built at width 32) |

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
   (Since modification 9 the floor there is -FLT_MAX, in the natural-log domain.)
9. **Bias and sink in the natural-log domain** (`flash_attn_func_gfx1201_aiw.py`,
   `generators/gen_sdpa_aiw.py`). Upstream keeps the scores base-2, `qk * scale * log2e`,
   and adds `bias * log2e` to them before subtracting the max. For any |bias| above
   2.3587e38 that product overflows f32: a bf16 or f32 `finfo.min` padding mask (the
   HF/transformers convention) became -inf, so a row masked that way on every key gave
   O = 0 and LSE = -inf where PyTorch gives mean(V) and an LSE near the bias, and a bias
   above 2.3587e38 gave NaN. The sink had the same overflow (`sink * log2e`, LSE NaN for a
   sink above 2.3587e38). A bias build now does what the RDNA kernel's modification 22
   does: `s = qk * scale + bias` (one FMA), the row max taken on s, and every exponent
   `exp2((s - m) * log2e)`, the difference formed first. Its m floor is -FLT_MAX rather
   than -3.40282e38, so an f32 `finfo.min` score sits above it; no softmax op carries
   `ninf`, and the ISA keeps the literal (`0xff7fffff`) as the loop's initial max. A
   sink build takes the sink's max, its term and the LSE in the same domain (a build
   without bias converts its base-2 m to m * ln2 there; the compiler fuses that into
   the subtract, which leaves a rounding residual of ~1e-7 relative in the rescale of O).
   The LSE is `m + log2(l) * ln2`. Builds with neither bias nor sink are byte-identical
   to before; all 32 bias and sink objects changed. Registers: the bf16 head_dim 128
   non-causal f32-mask object spills 4 VGPRs where it spilled 3 (its allowance rises
   to 4); the head_dim 256 bias objects spill 1-5 fewer; head_dim 64 f32-mask objects
   use 2-4 fewer VGPRs. The LSE of a row whose sum is NaN stays NaN (the zero test on l
   is a bit test, `v_cmp_class` for +0 in the ISA), PyTorch's answer.
10. **Padded rows filled, and padded lengths clamped** (`flash_attn_func_gfx1201_aiw.py`,
   `fmha_common_gfx1201.py`). Each object carries a second kernel,
   `flash_attn_aiw_padfill_kernel`, that writes a padded row (past its sequence's length,
   inside `Max_seqlen_q`) as O = 0 and LSE = -inf, hipDNN's reference contract (cuDNN >=
   9.14), sink builds included; hipDNN's native launches it by its symbol beside the main
   kernel for padded graphs only (never packed ones). The main kernel never stores such a
   row and is otherwise unchanged: the same stores inside it cost a VGPR in every form
   tried (in the epilogue, the prologue, or through the live stores), and at head_dim 128
   that VGPR cut occupancy, 19% slower. An INDIVIDUAL length (a padded sequence's) is
   clamped to `[0, max_seqlen]`, as hipDNN's GPU reference clamps it; every main kernel
   keeps its VGPR and spill counts.

Not modified, but checked: **MLA's head_dim 192 with a value head_dim of 128** builds
and runs as upstream wrote it. On gfx1201 against an f32 reference, causal and not and
under GQA, O is at bf16 precision and the LSE within 2e-6.

## Checked against the CDNA kernels' defects

Of the three defects found in the CDNA kernels (`../attention_cdna/README.md`,
modifications 21-23), two cannot occur here: the softmax scale multiplies the f32 scores
after the QK product rather than a bf16 Q before it, and the running max's finite floor
is already in the scaled domain, so no scale can overflow it.

The third, fast math licensing an -inf away as poison, does not occur here either.
The softmax operations carry explicit no-NaNs, not no-infinities, flags
(`fp_mode="noninf"`). Two broader settings stay on, as upstream ships them: the
ambient `fast_fp_math` hint, which includes no-infinities, and the backend's
`unsafe_fp_math`. They cannot reach the -inf because of how it is chosen. A row no
key reaches is detected by an *integer* test on the bit pattern of its sum `l`, and
the -inf is a literal in a select on that test. There is no float comparison against
infinity for a no-infinities flag to fold, and a poisoned value in the select's
other arm is never chosen. Checked on gfx1201, idle and under a concurrent GPU load
(the condition that exposed the CDNA defect):
- bottom-right causal with Sq > Skv, and windows that leave rows empty
- f16 and bf16, d64 and d128, with and without a sink

Across 6,192 empty rows per pass, every LSE is exactly -inf (the sink's value with a
sink) and O is 0.

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
- **Padding (per-batch lengths).** Every build reads `varlen_bits` at runtime, so
  hipDNN's padding (`padding_mask` with `seq_len_q`/`seq_len_kv`) needs no rebuild: the
  engine passes `VARLEN_SEQUSED_CACHE_SIDE` (0x04) on both sides, the INT32 length arrays
  as `seqinfo_q0`/`seqinfo_k0`, the batch as `num_seqlens` and the graph's extents as the
  maxima. Rows past a sequence's query length are not written. Bottom-right causal goes as
  the `WINDOW_BOTRIGHT` sentinel, resolved per sequence; other bottom-right bounds are one
  literal per launch, so the engine declines them under padding.
- **Dropout.** `ENABLE_DROPOUT` is a build axis (`has_dropout` objects, at `philox_width` 32
  pinned by the generator rather than read from the build host's GPU). The engine passes the
  seed and offset1 pointers (or offset2 by value), `idropout_p` computed in f32 on the host as
  AOTriton's kernel computes it (`philox.dropout_threshold` uses double and differs in about
  3e-8 of the elements; it is not used), and `1/(1 - p)`; the seed and offset outputs stay
  null. Its words fill each group of 4 keys in order, as AOTriton 0.15's `fast_philox`
  (0.11.1, which PyTorch 2.10 bundles, swaps words 1 and 2).

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).

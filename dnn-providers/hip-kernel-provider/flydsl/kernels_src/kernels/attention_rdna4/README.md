<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL attention for RDNA4 (gfx12)

Vendored kernel sources for a second prefill family of the `hipkernel:flydsl_sdpa_rdna`
engine on RDNA4. The RDNA3/RDNA4 kernels under `../attention_rdna/` stay the engine's
first family on both generations; this one adds AOTriton's FlyDSL forward kernel for
gfx12, which is faster on medium and long sequences.

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

## Checked against the CDNA kernels' defects

Of the three defects found in the CDNA kernels (`../attention_cdna/README.md`,
modifications 21-23), two cannot occur here: the softmax scale multiplies the f32 scores
after the QK product rather than a bf16 Q before it, and the running max's finite floor
is already in the scaled domain, so no scale can overflow it.

The third, fast math licensing an -inf away as poison, is narrowed but **not yet ruled
out**. The softmax operations carry explicit no-NaNs, not no-infinities, flags
(`fp_mode="noninf"`), and the LSE is computed with them. But two broader settings stay
on by default: the ambient `fast_fp_math` hint, which reaches operations that carry no
flags of their own and includes no-infinities, and the backend's `unsafe_fp_math`
for the whole compilation. Upstream notes that neither was ever measured on its own. The
provider's tests of rows no key reaches, run under a second GPU process, decide whether
they stay.

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).

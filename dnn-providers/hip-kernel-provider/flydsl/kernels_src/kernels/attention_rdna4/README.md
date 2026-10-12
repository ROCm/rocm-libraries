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

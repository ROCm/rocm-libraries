<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# K-outer transpose read on gfx1250 (wave32 WMMA) — case study

Extends the gfx950 studies
([wgrad](../../gfx950/conv_wgrad/wgrad_lds_layout_case_study.md),
[dgrad](../../gfx950/conv_dgrad/dgrad_lds_layout_case_study.md)) to the wave32
WMMA regime. The store side is unchanged and uninteresting here; the whole
content of this study is the **read** side, where the lane mapping is not a
port of the wave64 one.

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

## Why the wave64 formula does not port

Two regimes, pinned by `_LDS_K_OUTER_ARCH_WAVE = {"gfx950": 64, "gfx1250": 32}`
so a mismatched spec is rejected rather than emitting a formula the hardware
does not implement.

```
wave64 (gfx950, MFMA, ds_read_b64_tr_b16, 4 elements per lane)
    col  = mn_base + ((l % MN) // 16) * 16 + (l % 4) * 4
    row0 = k_base  + (l // MN) * n + ((l % 16) // 4)
    reads = n / 4

wave32 (gfx1250, WMMA, ds_load_tr16_b128, 8 elements per lane)
    col  = mn_base + ((l % 16) // 8) * 8
    row0 = k_base  + (l // 16) * n + (l % 8)
    reads = n / 8
```

gfx950's `((l % 16) // 4)` and `(l % 4) * 4` terms exist only because 64 lanes
over a 16- or 32-wide atom edge create groups *within* the free axis. With 32
lanes over a 16-wide edge the two lane groups split **K** instead, so those terms
have no counterpart.

gfx1250 has exactly one usable atom here — `16x16x32`, with
`a_frag_len == b_frag_len == 16` — so a fragment is **two** `ds_load_tr16_b128`
reads, not one. No new IR primitive was needed: `ds_read_tr16_b128` is
arch-agnostic in the IR and `core/isa/backend.py::ds_tr16_b128_spec` selects
`ds_load_tr16_b128` (`.v8bf16` / `.v8f16`) on gfx1250.

## Finding 1: the address a lane supplies is not the element it receives

This is the trap, and it cost a wrong-numerics debugging cycle.

The documented WMMA B layout
(`core/arch/target.py::_wmma_gfx1250_b_16x16x32`) says lane `l` holds column
`l % 16`, with fragment slot `i` at `k = (l // 16) * 16 + i`. The obvious
reading — "each lane reads a straight run of K at its own column" — gives

```
col  = mn_base + (l % 16)          # WRONG
row0 = k_base  + (l // 16) * n
```

That is wrong because `ds_load_tr16_b128` **transposes an 8x8 element block
within each group of 8 lanes**: the 8 lanes of a group each read 8 contiguous
elements, and lane `j` of the group receives element `j` from all 8 of those
runs. To *land* column `l % 16` in lane `l`, the group must address the 8-column
block containing it and each lane must supply a different K row of the run:

```
col  = mn_base + ((l % 16) // 8) * 8
row0 = k_base  + (l // 16) * n + (l % 8)
```

The wrong form is not a crash and not a near-miss. It reads a transposed
operand and produces numerically wrong output — while remaining perfectly
byte-identical across both engines.

## Finding 2: byte-identity cannot catch a wrong lane map

The two-engine gate proves Python and C++ emit the same bytes. It says nothing
about whether those bytes are *right*. Both engines carried the same wrong
formula and the gate stayed GREEN throughout.

What settled it was a **per-lane hardware probe**: fill LDS with an encoded
`(k, col)` value, issue the transpose read, and decode what each lane actually
received. One run answers the question that no amount of source reading did.
`examples/gfx1250/wmma_probe.py` is the existing precedent for this technique —
note its own docstring calls the gfx1250 lane maps a "hypothesis verified
empirically", so the layout this builds on carries the same caveat.

Generalisation worth keeping: **for any transpose-read intrinsic, probe the
per-lane distribution before trusting a derived address formula.** The
instruction's documentation describes what a lane ends up holding; the address
operand is a separate question.

## Finding 3: three guard defects surfaced while enabling this

Found while making the wave32 path reachable; all were latent on gfx950 too.

1. **The transpose-read arity guard was wave64-only.** dgrad rejected
   `b_per_lane % 4 != 0` for `ds_read_tr16_b64`'s 4 elements per lane. wave32
   returns 8, so fragment lengths of 4 or 12 passed the guard and left the
   `n // 8` read loop **empty**. Now derived from wave size, and added to wgrad,
   which flips both operands through the shared reader and had no guard at all.
2. **A 16-bit-dW epilogue rule sat outside its atomic guard.** "split_k atomic
   with dtype_d=... requires `epilogue='cshuffle'`" fired at `split_k=1`, where
   there are no packed atomics. The non-atomic direct-store path with 16-bit dW
   was therefore unreachable on **every** arch — and on gfx1250, where WMMA
   accepts only `'default'`, no valid wgrad spec with fp16/bf16 dW existed at
   all. On gfx950 it silently skipped every `epilogue=default` subTest in the
   wgrad pipeline sweeps.
3. **`pipeline="wavelet"` + `lds_k_outer` was reachable** from the sweep driver
   and would have written an M-outer tile into a K-outer allocation. See the
   dgrad study.

Defects 1 and 2 are the same species: a validation rule whose scope is wider
than its justification. Both presented as *tests passing while executing
nothing*, which is why the K-outer sweeps now carry an `_assert_ran()` backstop.

## Replay

From `platform/`, with `PYTHONPATH=$(pwd)/python`. Items marked **[gfx1250]**
need the hardware; the rest are CPU-only and run anywhere.

Correctness **[gfx1250]** — arch-aware, so on a wave32 box these select the
`16x16x32` atom and the `default` epilogue automatically:

```bash
python3 -m pytest tests/instances/test_conv_wgrad_correctness.py -q -rs -k k_outer
python3 -m pytest tests/instances/test_conv_dgrad_correctness.py -q -rs -k KOuter
```

The newly-reachable 16-bit-dW path from Finding 3.2 — read the `-rs` skip list
and confirm `epilogue=default` subTests **run** rather than skip:

```bash
python3 -m pytest tests/instances/test_conv_wgrad_correctness.py -q -rs \
    -k "pipeline_mem or pipeline_compv3 or pipeline_compv4 or pipeline_basic"
```

Emit-level check (CPU-only, runs on any box) — asserts the wave32 path actually
emits `ds_load_tr16_b128` with the right operand shapes, and that the wavelet
combination is rejected:

```bash
python3 -m pytest tests/instances/test_conv_dgrad_correctness.py -q \
    -k Gfx1250Emit
```

Byte identity, both flavors — necessary, **not sufficient**, per Finding 2:

```bash
export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only conv
ROCKE_LLVM_FLAVOR=llvm22 python3 tools/check_byte_identity.py --only conv
```

Step 0 lever sweep **[gfx1250]**:

```bash
python3 python/rocke/benchmark/benchmark_implicit_gemm_conv.py \
    --direction wgrad --arch gfx1250 --dtype bf16 \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 8 -c 128 -H 32 -W 32 -k 128 \
        -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -in_layout=NHWC" \
    --split-k 1 --jobs 32 --sample 0.05 --seed 0 --top 5 --warmup 3 --iters 10
```

WMMA wgrad requires `split_k=1` and the direct-store epilogue, so do not pass
`--split-k 0` here — the sweep will reject every combination.

## Environment note

gfx1250 needs ROCm >= 7.2. If the venv's torch bundles an older ROCm than the
system install, comgr resolution matters: a stale bundled comgr does not know
the target ISA and every compile dies with `set_isa: INVALID_ARGUMENT`, a long
way from its cause. Check before trusting a failure:

```bash
python3 -c "
import torch
import rocke.runtime.comgr as c
from rocke.runtime import runtime_coexistence as rc
print('torch rocm    :', rc._torch_rocm_version())
print('newest system :', rc._newest_rocm_root_version())
print('resolved comgr:', c.resolved_lib_path())
print('comgr vintage :', c.resolved_lib_rocm_version())
"
```

You should not need to set `ROCKE_COMGR_LIB`. If you do, that is a finding.

## Still open

- **Port the per-lane probe into this folder.** The probe that settled Finding 1
  currently lives outside the tree. It is the reusable artifact here — worth
  having as a rocke-DSL example before the next architecture.
- **A dgrad gfx1250 parity config** to match wgrad's config 15.
- **No gfx1250 dgrad dispatch candidate**; the wave32 dgrad path is reachable
  only through the sweep driver.

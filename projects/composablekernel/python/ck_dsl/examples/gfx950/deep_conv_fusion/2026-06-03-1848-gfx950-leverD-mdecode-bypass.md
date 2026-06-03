# gfx950 Deep Fusion — Lever-D: Eliminate the A-descriptor m-decode round-trip

Status snapshot 2026-06-03 18:48.

Follow-up to the lever-C rocprof note (`2026-06-03-1753`). That note concluded
the kernel is VALU-bound and that seg0 (the im2col coordinate prologue) was at
its codegen floor. That was wrong: seg0 contained a **redundant coordinate
round-trip** that this lever removes.

## The redundancy

The conv0 A load path was:

```text
m_index_fn(row)         -> m = global_h * Wo + global_w        (cheap: shr/and, then mul+add)
A_desc.offset(m=m, k=k) -> unmerge_magic(m -> n, ho, wo)       (~10 VALU: 2 magic divisions)
                           embed(ho,r -> hi); embed(wo,s -> wi)
```

`m_index_fn` already computes `global_h`/`global_w` cheaply (shift/mask, since
`conv_tile_w=8` is power-of-two), flattens them into `m`, and then the A
descriptor's leading `unmerge_magic(m -> n, ho, wo)` immediately un-flattens `m`
back into `(n, ho, wo)` via two magic divisions:

```text
÷Wo=3840  mult=0x11111112 shift=12
÷Ho=2160  mult=0xe573ac91 shift=12
```

A full magic unmerge step is `umul_hi + add + lshr` (quotient) + `mul + sub`
(remainder) ≈ 5 VALU, so the pair is ≈10 VALU per A coordinate — paid in the
hot K-loop, per A fragment. The flatten + unmerge are a pure no-op pair.

## The fix

`make_a_descriptor(p, decompose_m=False)` drops the leading
`unmerge_magic(m -> n, ho, wo)`; the descriptor's upper coords become
`(n, ho, wo, k)` directly. A new `a_mhw_index_fn` callback returns
`(n=0, ho=global_h, wo=global_w)` — the same shift/mask values `m_index_fn`
produced — and `a_descriptor` feeds them straight into the `embed` transforms.
N==1 for this problem so `n` is the constant 0.

Bit-identical by construction: with N=1, `unmerge_magic(m -> n,ho,wo)` of
`m = global_h*Wo + global_w` yields exactly `wo=global_w, ho=global_h, n=0`.
The `embed`/`pad` chain is unchanged, so the computed linear offset is identical.

Plumbing:

```text
conv_implicit_gemm.py
  make_a_descriptor(p, decompose_m=True)   # new flag; default keeps old behavior
  build_implicit_gemm_conv(..., a_mhw_index_fn=None)   # new optional hook
    A_desc = make_a_descriptor(p, decompose_m=(a_mhw_index_fn is None))
    a_descriptor: if a_mhw_index_fn -> offset(n,ho,wo,k); else offset(m,k)

deep_fused_conv_pool.py
  a_mhw_index_fn(b, row, grid) -> (const 0, global_h, global_w)   # new
  build_implicit_gemm_conv(..., a_mhw_index_fn=a_mhw_index_fn)
```

The k -> (r,s,c) decode is left alone: C=8 folds `÷8` to a single `lshr`
(mult=1/shift=3), and the surviving `÷3` (S, mult=0x55555556) is one cheap
decode. Only the m-decode was the redundant pair.

## ISA effect (llvm-objdump --mcpu=gfx950)

```text
metric                        before (lever C)   after (lever D)
----------------------------  -----------------  ---------------
v_mul_hi_u32 total            7                  3
v_mul_hi_u32 in seg0          6                  2
÷Wo magic const 0x11111112    present            GONE
÷Ho magic const 0xe573ac91    present            GONE
```

The two remaining seg0 `v_mul_hi_u32` are the `÷3` S-decode (unavoidable). The
one outside seg0 is the D (output) descriptor's m-unmerge, left intact (epilogue
addressing, runs once per output — not hot).

## Wall clock

```text
config                 wall clock       useful TFLOP/s
---------------------  ---------------  --------------
lever C (m-flatten)    0.184 ms         277
lever D (m-bypass)     0.1777-0.1792 ms 284-287   (~+2.7%)
```

verify: `max_abs_diff=0.00195312 bad=0/49766400` — bit-identical to lever C.
Stable across 4 runs (0.1777-0.1792 ms).

The ~2.7% gain matches the prediction: seg0 is one of several VALU regions in a
~63%-VALUBusy kernel, so removing ~10 VALU/coord from one block helps modestly.
It is, however, the first genuinely reducible chunk of the VALU floor we had
written off as irreducible — the prior "seg0 is codegen-optimal" claim was the
miss.

## Reproduce

```text
verify + bench:
  HIP_VISIBLE_DEVICES=1 python3 -m ck_dsl.examples.gfx950.deep_conv_fusion.compare_pool_tile_configs

ISA check:
  compile spec (pool_tile=4x4 tk32 tn32 warp 2x1) -> hsaco
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 best.hsaco | grep -c v_mul_hi_u32
  grep -c '0x11111112\|0xe573ac91'   # -> 0 after lever D
```

# Case study: gfx1250 fp16 RCR GEMM — tensor-DMA + prefetch (~1.40x)

Replayable record of the change that took the gfx1250 fp16 RCR candidate from
the step-0 sweep winner to ~1.40x of it — ~0.97 of the **LDS-fed** WMMA roofline
from tensor-DMA and prefetch, then a final ~1.02x from register blocking.

Per `platform/AGENTS.md` §Compliance this file carries **ratios only** — no
absolute throughput. Absolute numbers live in the protected tracker.

Arch facts, bit layouts and the levers are in
[`arch/gfx1250.md`](../../../../dsl_docs/optimization/arch/gfx1250.md) §21.10.
This file is the *narrative*: what was tried, what the measurement said, and
which conclusions had to be thrown away.

## Result

| step | relative | % of LDS-fed ceiling |
|---|---|---|
| step-0 sweep winner (`128x128x64`, cooperative copy) | 1.00 | ~0.71 |
| `+ tdm_lds` alone | 0.99 | — |
| `+ tdm_prefetch` (depth 2) | 1.09 | ~0.78 |
| `+ 256x256x64 w4x4` geometry | 1.26 | ~0.90 |
| `+ tdm_prefetch_depth=3` | 1.33 | ~0.94 |
| `+ tdm_prefetch_depth=4` | 1.37 | ~0.97 |
| `+ w4x2` register blocking (depth 3) | **1.40** | — |

> **The winning row is not what ships.** Every measurement in this file was
> taken from artifacts built during the tuning run, and those artifacts still
> run correctly. Rebuilt from the current tree, the `w4x2` / depth-3
> configuration — and every other 8-wave TDM geometry, plus `w2x2` at depth 3 —
> **faults at launch** with an HSA aperture violation. The sources, the three
> available ROCm installs and the build are deterministic and identical; the
> cause is not yet identified.
>
> So `dispatch/gemm/fp16_rcr.py` registers the nearest configuration verified to
> build and validate from a fresh tree: **`w2x2` at depth 2**. By the component
> ratios in this file that is about `0.98x` for the warp grid and ~10% for the
> depth; the direct A/B re-measurement is still outstanding. Read the table
> below as the tuning record and the target to restore — not as a description
> of the shipped candidate.

> **The magnitudes need re-validating.** These ratios were taken in an earlier
> measurement environment. Ratios are supposed to survive a change of
> environment, but at least one lever demonstrably did not: the split barrier
> (not in this table, and not enabled by any registered candidate) measured
> `1.11x` at `K=65536` then and `1.008x` on re-measurement. A lever that hides
> latency stops paying once the kernel is no longer bound by what it was
> hiding.
>
> So read every row as measured-at-the-time. The *ordering* of the ladder is
> likely still right — each step addresses a different bottleneck — but the
> magnitudes should not be quoted without a re-run, and the
> `% of LDS-fed ceiling` column is normalized against a ceiling that has since
> been re-measured.

Registered as `universal_gemm_fp16_gfx1250_wmma_tdm` (priority 5) in
`dispatch/gemm/fp16_rcr.py` — see the note above for why it registers `w2x2` /
depth 2 rather than the `w4x2` / depth 3 this study measures. The `128x128`
entry keeps `tdm_lds` + `tdm_prefetch` at depth 2 and still serves shapes not
divisible by 256.

## Replay

```bash
cd rocke/platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only gemm        # must stay GREEN 21/21

# build what the dispatcher actually returns, then verify on a GPU host
python3 -m rocke.examples.gfx1250.gemm.build_candidates \
    --candidates <candidates.json> --out <out> --shape 4096,4096,4096
python3 -m rocke.run_manifest <hsaco> <manifest> --verify --shape 512,512,512
```

Measurement protocol that the numbers above depend on: idle device (abort if any
foreign process holds `/dev/kfd`), all variants **interleaved** within one run so
drift hits them equally, >= 4 attempts, and the reference re-measured alongside
so each result carries its own evidence of validity.

## What had to be built first

`tensor_load_to_lds` needs the operand's raw 64-bit address, but `A`/`B` are
`PtrType` params and rocke exposed `ptrtoint` only for `addrspace(3)`. So this
needed a new cross-engine IR op, `global_addr_of` — the `addrspace(1)` peer of
`smem_addr_of`, seven touchpoints per `dsl_docs/development/extending.md` §1.

`helpers/tdm.py` packs the five D# descriptor groups. It is pure composition
over existing IR ops, hence Python-only, matching `split_k` / `pipeline` /
`sparse_iter`.

## Five things that were wrong, and what corrected them

Each of these survived reasoning and died to a measurement. They are the actual
content of this case study.

**1. A full-width shift, invisible to every static check.**
`stride[0]` splits at bit 32; for a *runtime* stride that emitted
`lshr i32 %K, 32` — poison in LLVM. Garbage reached the descriptor's stride
field and the DMA engine faulted. It compiled, assembled, and passed a
bit-level packing test that only checked constants. Caught by the first device
launch. Now guarded: SSA operands are i32, so a split at >= 32 returns literal 0.

**2. Every wave issued the same descriptor.**
The descriptor covers the whole tile, so all 4 waves re-fetched it — 4x the
global traffic. Harmless for correctness, which is why verification passed it,
and it would have made the first measurement meaningless. One wave issues now,
inside `scf_if(warp_id == 0)`; the existing barrier publishes.

**3. The WMMA read path had no double-buffer support at all.**
`emit_mfma_phase` early-returns for `family == "wmma"` into `_emit_wmma_phase`,
which took no `lds_parity`. Writes alternated halves while reads always took
half 0. A first "fix" to the parity gate in `emit_mfma_phase` was **dead code**
on this path. Localised by bisecting on K-tile count: 1 tile correct (parity is
always 0), >= 2 wrong.

**4. "TDM is a loss" — true, and the wrong conclusion.**
Single-buffered `tdm_lds` measured 0.99 and the fill side looked exhausted. It
was measuring the wrong configuration: the descriptor was issued and immediately
waited on, so nothing overlapped. ck_tile's TDM pipelines declare
`PrefetchStages = 2` for exactly this reason. Never evaluate `tdm_lds` without
`tdm_prefetch`.

**5. "The pad is free now, so geometry must be re-tuned" — right action, wrong
reason.** TDM makes `lds_k_pad` free to *apply*, which is not the same as making
it unnecessary: the pad buys bank-conflict avoidance on `ds_read`, which TDM
does not touch, and `ds_read` is now *more* dominant. Measured, pad 8 stayed a
sharp optimum across seven geometries and four depths (pad 0 → 0.36 at
`128x128`). The re-sweep was still worth running — it found the 256x256 tile —
but not for the reason it was started.

## The gate that was hiding the best configuration

`256x256` was rejected before it could be measured, by a double-buffer
affordability test that was wrong twice over: it sized the tile **unpadded**,
and it demanded headroom for a second workgroup per CU that VGPR pressure
already prevents. Fixing both unlocked the two fastest configurations.

A caution on reading resource counts: the tuned kernel allocates 256 VGPR with
**50 spills**, not the spill-free allocation an earlier profiling pass reported
— that pass read the neighbouring depth-3 artifact. Confirm which artifact a
resource number came from before drawing a conclusion from it; the two differ
only in a trait, and the filenames do not say so.

Related: `arch_specs.json` carries `lds_capacity_bytes = 163840` (per CU), but
HIP reports `sharedMemPerBlock = 327680` — a WGP is two CUs sharing one LDS
block and a workgroup may take all of it. Device-confirmed by launching a
184320 B tile. Going WGP-exclusive is legal but did **not** pay here.

## A dead end worth not repeating

`global_load_tr16_b128` looks like the way to feed WMMA from global and skip LDS
entirely. It assembles on gfx1250 and returns data to only 4 of 32 lanes under
every address pattern. CK gates its only consumer on `__gfx120__` (RDNA4), not
`__gfx125__`; compiling the maximally-favourable both-operands-transposed
instance for gfx1250 yields zero `global_load_tr`. The LDS-side
`ds_load_tr16_b128` *is* supported — but ck_tile enables it only for
`ALayout == ColumnMajor` / `BLayout == RowMajor`, which is the complement of
RCR, so it is not a lever for this kernel either. Details in §21.3.

## Register blocking: the last 2%, and why it is only 2%

Once the fill side was solved, LDS reads per WMMA became the cost, and that is
fixed by the per-warp atom grid at `(m+n)*2/(m*n)`. Shrinking the warp count
enlarges the grid and improves the ratio. It works, barely, and the shape of the
result is more useful than the gain:

- 1.00 → 0.75 (`w4x4` → `w4x2`) is worth **1.02x**.
- Pushing further **loses**: 0.625 measures 0.96x, 0.50 measures 0.98x. The
  ratio keeps improving while throughput turns over, because the >255-index
  VGPR latency and the lost occupancy overtake the saved traffic.
- **Orientation is worth ~10%** and the ratio formula cannot see it: `w4x2` and
  `w2x4` are identical in atom count, accumulator size and ratio, yet measure
  1.02 and 0.93. Spills track it (48 vs 154).
- **Spill count is the wrong objective.** The only nearby zero-spill
  configuration is ~10% slower than the 48-spill winner.

The roofline said a 0.50 ratio was worth 1.69x at the operand-supply level. It
was — as a *ceiling*. Register blocking cannot collect it, because the register
pressure needed to reach 0.50 costs more than the traffic it saves. A ceiling
measured in isolation does not tell you the route is reachable.

## Where the remaining headroom is

At ~0.97 of the LDS-fed ceiling this axis is finished. The wait-ablation
diagnostic bounds what is left to ~1.10, and depth 5 exceeds the LDS ceiling, so
deeper pipelining cannot reach it.

What remains is the **1.87x LDS-fed vs register-fed gap** measured by
`roofline_probe.py`. Register blocking has now been tried against it and tops
out at ~1.02x (above), so the remaining routes are sourcing an operand outside
LDS entirely — `roofline_probe.py`'s `build_wmma_hybrid_kernel` measures that
ceiling at ~1.69x — or wave specialization (the `wavelet` pipeline, not
reachable from `UniversalGemmSpec` today).

Register pressure bounds any such attempt: the tuned kernel already allocates
512 VGPR with 48 spills, and every configuration that pushed past it measured
slower.

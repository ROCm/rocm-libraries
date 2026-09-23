# CK DSL Optimization Runbook — Per-Architecture References

> Specializes the [CK DSL Optimization Runbook](../optimization_runbook.md).

The [optimization runbook](../optimization_runbook.md) is
architecture-agnostic: every principle and lever applies to any AMDGPU
CDNA target. The **concrete per-architecture facts** — MFMA atom
catalog, LDS size / banks / conflict periods, cross-lane primitives,
register / occupancy caps, chiplet / XCD layout, buffer-descriptor
flags, fp8 / quantization support, compiler caveats, and the default
ISA — live in the per-arch files indexed below.

When the base runbook hits an arch-mixed spot (LDS specifics in
§6.3/§6.4, MFMA atom selection in §7.1/§7.4, compiler caveats in §10,
scheduling in §8, ISA inspection in §11, autotuning in §12), it keeps
the **general principle inline** and defers the concrete arch fact to a
single hub — **§21 Target Architecture Reference** in the runbook. Each
of those spots links to §21 (not to the arch files directly), and §21 is
the one place that lists the per-arch files indexed below. That is why
adding an architecture touches only §21 and the index here, never the
inline breadcrumbs.

## Index

| Architecture | GPU(s) | Reference |
|---|---|---|
| gfx950 (CDNA4) | MI350X / MI355X (DSL default target) | [arch/gfx950.md](gfx950.md) |
| gfx942 (CDNA3) | MI300X / MI325X | [arch/gfx942.md](gfx942.md) |

## Arch-reference template (the 10-subsection skeleton)

Every arch file mirrors the same 10-subsection skeleton (the structure
the gfx950 reference established as `21.x`). Subsection numbers are kept
as `21.x` across all arch files so the base runbook's cross-references
resolve uniformly regardless of which arch you are reading.

| # | Subsection | Content |
|---|---|---|
| 21.1 | MFMA Atom Catalog | Atoms available on the arch: A/B/C dtype, M×N×K, K-pack and chained-MFMA lane-layout notes, fp8/bf8/scaled/MX/i8/i4 coverage |
| 21.2 | LDS specifics | LDS per CU, bank count, preferred swizzle (§6.4a), transpose-read availability, per-opcode bank-conflict periods, read/write asymmetry |
| 21.3 | Cross-lane primitives | `permlane`/`ds_bpermute`/`ds_swizzle`/`ds_read_tr*`/`s_setprio`/`sched_barrier`/`s_barrier`/`iglp_opt` availability and DSL surface |
| 21.4 | Register / occupancy | SIMDs/CU, waves/SIMD, waves/CU, VGPR/AGPR/SGPR caps, allocation granularity, threads/CTA, wave size |
| 21.5 | Chiplet / XCD | XCDs per package, `NUM_XCDS_*` constants, grid-swizzle helper, default chiplet swizzle params |
| 21.6 | Buffer descriptor (AMDGPU) | OOB-safe DW3 flag, `buffer_rsrc` surface, async DRAM→LDS dword constraints + write contract, encoded `s_waitcnt` constants |
| 21.7 | FP8 / quantization | Supported low-precision dtypes (range, mantissa, native-MFMA), MX / block-scale / codebook surfaces |
| 21.8 | Compiler caveats | Arch-specific backend hazards (dropped/ICE'd intrinsics, risky flags), what the recommended flag stack actually buys |
| 21.9 | Default ISA target | `compile_kernel` default ISA, datalayout, how to override to this arch, atom-validity check |
| 21.10 | Pointers to deeper material | Arch-specific empirical studies, skill briefs, case-study docs |

Each arch file starts with the standard back-link header:

```markdown
> Specializes the [CK DSL Optimization Runbook](../optimization_runbook.md).
> Sibling arches: see [arch/README.md](README.md).
```

## How to add a new architecture

1. **Copy the skeleton.** Start from [arch/gfx950.md](gfx950.md) and
   keep all ten `21.x` subsection headers; replace the body of each
   with the new arch's facts. Lead with the new arch's facts as
   first-class — do not frame an arch file as a contrast/diff against
   another arch.
2. **Fill the 10 sections** from the vendor spec sheet plus the DSL's
   own per-arch tables (`helpers/atoms.py`, `helpers/grid.py`,
   `core/lower_llvm.py`, measured LDS / occupancy studies).
3. **Register it here** — add a row to the Index table above.
4. **Register it in the base runbook's §21 hub** — add the new file to
   the "Pick your target" list in
   [§21 Target Architecture Reference](../optimization_runbook.md#21-target-architecture-reference).
   That section is the *only* spot in the runbook that links arch files
   directly; the inline "architecture reference" breadcrumbs throughout
   §6/§7/§8/§10/§11/§12 already point at §21, so they need no per-arch
   edits.
5. **Keep each arch file single-arch.** A new arch file's tables should
   carry only that arch's values — do not replicate the multi-arch
   comparison columns. (The gfx950 reference's §21.2 / §21.9 still list
   `gfx942` / `gfx90a` columns for historical continuity from the
   original monolithic runbook; treat each arch's own file as the
   source of truth, not those legacy columns.)

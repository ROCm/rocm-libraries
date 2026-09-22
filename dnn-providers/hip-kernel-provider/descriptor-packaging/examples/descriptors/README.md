# Example production descriptor tree

A minimal but **real** authored source root for `hkp_pack`. Both producers are exercised
end to end: the hip half compiles a `.cpp` with `hipcc`, the rocKE half lowers a real
rocKE builder through comgr. Placeholder shapes, real code paths.

This is a **test fixture** tree, not the default production source root. That default is
`src/engines/kernel_ingestor_engine/descriptors/`, whose README states the gate and the
rules a bundle authored there must meet. The Linux superbuild CI lane points
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` at this tree, so the ordinary production rules
pack it and install the output into that build's prefix.

`test_hkp_pack_layout.py` packs this tree directly, which is what makes its layout
assertion strict, and `test_desk_check_invariants.py` reads its bundles as real authored
input: changing anything here changes what those tests pin.

## Layout

```
descriptors/                      <- the ONE source root
├── hip/
│   └── pointwise_add/            hip producer: kdp + generics + PointwiseAdd.cpp
│       └── shared.umd.json
└── rocKE/
    └── gfx942_tiled_attention/   rocKE producer: kdp + generics, no local sources
        └── shared.umd.json
```

Child folders scope the content; producer selection is per-UKD on `kernel_source.kind`,
never per-folder. The authored subpath is preserved verbatim into the staged and
installed trees, so the shipped layout mirrors this one:

```
arch_content/hip-kernel-provider/gfx942/
├── kpack/hip_kernel_provider_gfx942.kpack     <- one per arch, at the arch root
├── hip/pointwise_add/...
└── rocKE/gfx942_tiled_attention/...
```

**`shared.umd.json` appears in both child folders on purpose.** The duplicate filename
keeps this tree a standing check that packing is path-preserving: a flat packer drops one.

## Authoring rules worth knowing

**`source` means different things per producer.** For `kind: "hip"` it is a file path
resolved **relative to the descriptor that names it** — a sibling `.cpp`, not a path from
the root. There is no root-relative fallback, so a miss is an error rather than a silent
bind to a same-named file elsewhere; share one `.cpp` between sibling folders by saying
so: `"../shared/Kernel.cpp"`. For `kind: "rocke"` it is a **dotted Python module path
resolved through the importable `kernels` package**, not a file under this root:
`kernels/gfx942/attention_tiled_2d.py` comes from the installed rocKE wheel, which is why
the rocKE folder carries no sources.

**`builder` names a function taking `(spec, *, arch)`.** A builder with extra
keyword-only parameters is rejected rather than packed, because a descriptor cannot
supply them and they would be frozen at their defaults. `spec` is constructed into the
builder's own spec dataclass, so its fields and their validation are the builder's.

**The launch symbol is never authored.** It is captured from the compiled artifact.

**`arch` filters which shard a descriptor ships in.** It does not select a builder:
naming `gfx942` does not make a gfx950 builder produce gfx942 code.

**`library` is relative to the descriptor that declared it**, and the archive is one per
arch at the arch root, so a descriptor in a child folder climbs back out to reach it
(`../../kpack/...`). The runtime joins `library` onto the descriptor's own directory but
bounds the result by the descriptor TREE. An arch-root-relative value is correct only for
a descriptor sitting flat at the arch root, and silently wrong for every nested one.

## Why this rocKE builder

`build_unified_attention_2d_tiled` rather than `build_attention_dense`. Both are accepted
— no builder in this corpus is refused, so the packer's rejection shapes are covered
synthetically in `tests/test_hkp_pack_producer_guards.py`.

The tiled builder's spec is compile-time shape only — head size, KV block size, head
counts, dtype, feature flags — with sequence count and lengths arriving at runtime through
`cu_q` and the block tables, so one authored descriptor covers every problem size.
`AttentionDenseSpec` bakes `batch`, `seqlen_q` and `seqlen_kv` in as constants, so a
descriptor naming it pins its kernel to one problem shape.

**The rocKE half borrows the pointwise pack's native symbols, which bounds what this tree
proves.** A descriptor only resolves to something a compiled native pack registered, and
that is `hipkernel.pointwise.*` and `hipkernel.conv.*`; there is no rocKE/attention pack.
This tree therefore proves the **packaging** path for rocKE — authored descriptor →
comgr-lowered kernel → kpack archive → install layout, with per-UKD `kind` dispatch
exercised for real — and not rocKE-specific runtime dispatch.

The descriptors here are authored against the schema the C++ loader enforces, modelled on
`src/engines/kernel_ingestor_engine/test_descriptors/`. Do not model them on
`descriptor-packaging/tests/fixtures/`: that is packer-only test data which never passes
through `DescriptorLoader.hpp`, so a tree copied from it can pack cleanly and still fail
to load.

gfx942 rather than gfx950 because `hipdnn-linux-superbuild` — the lane that can gate
this — builds gfx942.

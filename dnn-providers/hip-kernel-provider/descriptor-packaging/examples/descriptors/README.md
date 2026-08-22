# Example production descriptor tree

A minimal but **real** authored source root for `hkp_pack`. Both producers are
exercised end to end: the hip half compiles a `.cpp` with `hipcc`, the rocKE half
lowers a real rocKE builder through comgr. Placeholder shapes, real code paths.

This tree exists because nothing else drives the production path. Before it, the
production source-root variables were empty in every preset and CI lane, which
is how a silent-empty install (review 3.9) and a silent descriptor drop (review
2.1) both survived unnoticed.

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

There is exactly one root. Child folders scope the content; producer selection is
per-UKD on `kernel_source.kind`, never per-folder. The authored subpath is
preserved verbatim into the staged and installed trees, so the shipped layout
mirrors this one:

```
arch_content/hip-kernel-provider/gfx942/
├── kpack/hip_kernel_provider_gfx942.kpack     <- one per arch, at the arch root
├── hip/pointwise_add/...
└── rocKE/gfx942_tiled_attention/...
```

**`shared.umd.json` appears in both child folders on purpose.** The filename is
deliberately reused so this tree is a standing regression test for review 2.1: a
flat, non-path-preserving packer silently drops one of the two. The in-tree
ingestor descriptor set does the same thing accidentally with
`kernel_dtype_matches_graph.umd.json`.

## Authoring rules worth knowing

**`source` means different things per producer.** For `kind: "hip"` it is a file
path resolved **relative to the descriptor that names it** — a sibling `.cpp`,
not a path from the root. There is no root-relative fallback: a miss is an error,
because falling back would turn a typo into a silent bind to a same-named file
elsewhere in the tree. To share one `.cpp` between sibling folders, say so:
`"../shared/Kernel.cpp"`.

For `kind: "rocke"` it is a **dotted Python module path resolved through the
importable `kernels` package** — *not* a file under this root. `kernels/gfx942/
attention_tiled_2d.py` is found via the installed rocKE wheel, which is why the
rocKE folder carries no sources. This is the single biggest clarity trap in the
format.

**`builder` names a function taking `(spec, *, arch)`.** Nothing else. A builder
with extra keyword-only parameters is rejected rather than packed, because a
descriptor cannot supply them and they would be silently frozen at their
defaults. `spec` is constructed into the builder's own spec dataclass, so its
fields and their validation are the builder's, not ours.

**The launch symbol is never authored.** It is captured from the compiled
artifact. Authoring it would let the descriptor disagree with the kernel.

**`arch` filters which shard a descriptor ships in.** It does not select a
builder: naming `gfx942` does not make a gfx950 builder produce gfx942 code.

**`library` is arch-root-relative.** One `kpack/` per arch root; descriptor
nesting does not move it.

## Why this rocKE builder

`build_unified_attention_2d_tiled` rather than `build_attention_dense`: the
latter takes a keyword-only `tuning: Gfx942DenseTuning` that no descriptor can
set, so the packer refuses it by design. That refusal has its own regression test
(`test_real_gfx942_attention_dense_is_refused`); this tree covers the happy path.

**The rocKE half borrows the pointwise pack's native symbols, and that bounds what
this tree proves.** A descriptor only resolves to something a compiled native pack
registered, and today that is `hipkernel.pointwise.*` and `hipkernel.conv.*` — there
is no rocKE/attention pack. So this tree proves the **packaging** path for rocKE:
authored descriptor → comgr-lowered kernel → kpack archive → install layout, with the
per-UKD `kind` dispatch exercised for real. It does **not** prove a rocKE-specific
runtime dispatch; that needs a native pack nobody has written yet. Writing one is the
next step toward a true rocKE end-to-end.

The descriptors here are authored against the schema the C++ loader enforces, taken
from `src/integration_tests/kernel_ingestor_engine/fixtures/packaged/` rather than
from `descriptor-packaging/tests/fixtures/`. The latter is packer-only test data that
never passes through `DescriptorLoader.hpp`, and modelling this tree on it once
produced a tree that packed cleanly and could not be loaded at all.

gfx942 rather than gfx950 because `hipdnn-linux-superbuild` — the lane that can
gate this — builds gfx942.

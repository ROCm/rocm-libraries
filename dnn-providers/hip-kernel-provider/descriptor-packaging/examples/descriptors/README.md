# Example production descriptor tree

A minimal but **real** authored source root for `hkp_pack`. Both producers are
exercised end to end: the hip half compiles a `.cpp` with `hipcc`, the rocKE half
lowers a real rocKE builder through comgr. Placeholder shapes, real code paths.

This tree drives the production packaging path with fixtures. Without a source root
pointed here, the pack step uses the provider's shipped descriptor root instead, and
skips when nothing there declares an architecture the build packs for.

## Disposition

| Set | Verdict |
|---|---|
| `hip/pointwise_add/` | not shipped — CI production-path exercise, and layout fixture |
| `rocKE/gfx942_tiled_attention/` | not shipped — CI production-path exercise, and layout fixture |

Neither set reaches a product build: `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` defaults to
the provider's shipped descriptor root, so nothing points at this tree unless a build
overrides it, and no install rule copies it. Two things do use it.
`hipdnn-superbuild-ci.yml` points the production source root here in the Linux lane, which
runs the production packaging path end to end on these fixtures.
`tests/rocke/test_hkp_pack_layout_rocke.py` packs the whole tree and
`tests/test_hkp_pack_layout.py` packs its `hip/` subtree, which is what makes their layout
assertions strict: changing anything here changes what those tests pin.

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

**`shared.umd.json` appears in both child folders on purpose.** Reusing the
filename keeps this tree a standing check that packing is path-preserving: a
flat packer silently drops one of the two.

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

**For rocKE, the launch symbol is never authored.** It is captured from the
compiled artifact. Authoring it would let the descriptor disagree with the kernel.

**For `kind: "hsaco"`, `file` names a prebuilt code object and `symbol` IS
authored.** `file` resolves relative to the descriptor that names it, must stay
inside the source root, and has no root-relative fallback, like a hip `source`.
The object is packed as-is, with no compile. This tree contains no hsaco example.
Each hsaco UKD needs a non-empty `arch`: [rules](../../README.md#hsaco-kernels-and-arch).

**`arch` filters which shard a descriptor ships in.** It does not select a
builder: naming `gfx942` does not make a gfx950 builder produce gfx942 code. For generic
targets, see [Generic GPU targets](../../README.md#generic-gpu-targets).

**`library` is relative to the descriptor that declared it**, and the archive is
one per arch at the arch root — so a descriptor in a child folder climbs back out
to reach it (`../../kpack/...`). Both halves matter: the runtime joins `library`
onto the descriptor's own directory, but it bounds the result by the descriptor
TREE, not by that directory. Writing the value arch-root-relative instead is
correct only for a descriptor sitting flat at the arch root, and silently wrong
for every nested one.

**A Universal Engine Descriptor (UED) `graph_match` holds exactly one arm.** The
optional `graph_match` object carries either `native`, a nonempty symbol the
provider registers, or `nodes`, a declarative graph pattern (RFC 0020 §4.3).
A pattern resolves against the op-schema registry generated from the hipDNN
graph schemas (`projects/hipdnn/flatbuffers_sdk/op_schema_registry.json`): every
opcode, operand name, and result name must exist there, and `?` may mark only an
operand the registry declares optional. `hkp_pack` applies the same checks as the
runtime, so a pattern the runtime would refuse fails the pack. Unknown fields are
rejected. Extensions use `x-` or `_` prefixes or the key `provenance`; they do not
declare bindings. The JSON Schema is
`projects/hipdnn/plugin_sdk/schemas/universal_engine_descriptor.schema.json`.

**Preserve producer identity and generic descriptor bytes.** Generic descriptors
must be byte-identical across architecture shards. A native UED names a symbol,
not binding keys, and native code defines the tensor binding names, so renaming
those bindings does not require UED edits. A `nodes` pattern binds the variables
it names.

**Use a matching provider and descriptor package.** The runtime skips malformed or
unsupported files individually, not whole packages. Identical accepted JSON
definitions count as one. Conflicting accepted definitions within one root
invalidate that UUID; across roots, the first accepted definition wins. If the
winner fails native-symbol checks, the loader does not try a later root. A pack
can still resolve a UUID in another root if an earlier malformed file was skipped.

`HIPDNN_DESCRIPTOR_RUNTIME_DIR` adds a tree without overriding installed
definitions. For a replacement tree, set `HIPDNN_DESCRIPTOR_DIR` and use a
matching provider. Start a fresh process to load descriptor or environment changes;
discovery does not reload them.

## Why this rocKE builder

`build_unified_attention_2d_tiled` rather than `build_attention_dense`. Both
carry the `(spec, *, arch)` signature the packer requires, and each has a
regression test holding it there (`test_real_gfx942_tiled_2d_is_accepted`,
`test_real_gfx942_attention_dense_is_accepted`), so the choice is about which
one models a shipped descriptor. The tiled builder's spec is compile-time
shape only — head size, KV block size, head counts, dtype, feature flags — with
sequence count and lengths arriving at runtime through `cu_q` and the block
tables, so one authored descriptor covers every problem size. `AttentionDenseSpec`
bakes `batch`, `seqlen_q` and `seqlen_kv` in as constants, so a descriptor naming
it pins its kernel to one exact problem shape and the tree would need another
descriptor for the next one.

**The rocKE half borrows the pointwise pack's native symbols, and that bounds what
this tree proves.** A descriptor only resolves to something a compiled native pack
registered, and today that is `hipkernel.pointwise.*` and `hipkernel.conv.*` — there
is no rocKE/attention pack. So this tree proves the **packaging** path for rocKE:
authored descriptor → comgr-lowered kernel → kpack archive → install layout, with the
per-UKD `kind` dispatch exercised for real. It does **not** prove a rocKE-specific
runtime dispatch; that needs a native pack nobody has written yet. Writing one is the
next step toward a true rocKE end-to-end.

The descriptors here are authored against the schema the C++ loader enforces,
modelled on `src/engines/kernel_ingestor_engine/test_descriptors/`.
Do not model them on `descriptor-packaging/tests/fixtures/`: that is packer-only
test data which never passes through `DescriptorLoader.hpp`, so a tree copied
from it can pack cleanly and still fail to load.

gfx942 rather than gfx950 because `hipdnn-linux-superbuild` — the lane that can
gate this — builds gfx942.

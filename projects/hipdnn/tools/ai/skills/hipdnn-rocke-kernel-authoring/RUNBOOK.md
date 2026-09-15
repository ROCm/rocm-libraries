# Runbook: author a rocKE kernel for a hipDNN graph

The **only ordered workflow**. [SKILL.md](SKILL.md) owns entry and completion; the linked
pages own contracts. Each gate requires a current observation; a previous session's log, a
sibling kernel's result or a proposed command is not evidence.

Every artifact a later gate needs is produced by an earlier step of this sequence. Do not
assume an environment, an interpreter or a device is already in place.

## Paths and interpreters

Resolve absolute paths first. These are explicit arguments, not implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"
ROCKE="$PROVIDER/rocke"
GEN="$REPO/projects/hipdnn/tools/IngestorGenerator"
PY=/absolute/path/to/rocke-venv/bin/python    # produced by step 3, not assumed
WORK=/absolute/path/to/scratch                # drafts, packs, logs — outside product source
GRAPH=/absolute/path/to/graph.json-or-.bin
ARCH=gfx950
```

Keep drafts, packs and logs under the workspace's scratch and evidence roots, not in
product source, until the kernel is handed to integration. Do not run GPU work on a login
host.

## 1. Entry

Record [SKILL.md](SKILL.md)'s entry contract: the graph and its form, the target
architecture and whether a device of that architecture is reachable, the kernel family,
the independent numeric reference, and the dtypes, layouts and shape envelope requested.

State whether the request is "serve this graph" or "serve this operation family". The
second is a scope decision, not an assumption you may make.

**Gate:** a graph in hand, a named architecture from rocKE's supported set, a named
reachable device *or* an explicit statement that step 7 will be blocked, a named reference,
and a stated shape envelope. A missing item is a blocked gate, not a default.

## 2. Check whether a builder already exists

Roughly thirty builders ship under
`dnn-providers/hip-kernel-provider/rocke/library/kernels/`. Read the family index
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/instances/index.md:1-10`) and
the per-family page — for attention, the variant-to-file map at
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/instances/attention.md:8-45`.

If a shipped builder serves the graph, **stop authoring.** The work is mining and
descriptor generation: go to
[rocke-mining.md](../hipdnn-ingestor-engine/rocke-mining.md), and return here only for
[handover-contract.md](handover-contract.md). Writing a second builder for a problem an
existing one solves is the most expensive mistake available at this point.

If an existing builder *nearly* serves it, extending its spec dataclass is preferred over a
new module — but adding a field is a baked/runtime decision, so read
[handover-contract.md](handover-contract.md) §2 before adding one.

**Gate:** the family page read, and a written statement naming either the existing builder
by path with why it does not serve the graph, or the family as having no candidate. "I did
not find one" without naming where you looked does not discharge this.

## 3. Stand up the environment

Nothing after this step is observable without an interpreter that imports both `rocke` and
`kernels`. Build it now.

Either configure with `-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON` and use the resulting
`build/rocke-pyenv/bin/python`, or create the venv by hand with the two editable installs
in the order [build-and-environment.md](build-and-environment.md) gives. Record which
route you took.

Record `ROCKE_BACKEND` and the resolved `ROCKE_LLVM_FLAVOR` now, not later: both change
results, and `ROCKE_BACKEND=cpp` falls back silently when `rocke_engine` is not built —
[build-and-environment.md](build-and-environment.md).

```bash
"$PY" -c "import rocke, kernels; print(rocke.__file__); print(kernels.__file__)"
```

**Gate:** that command exits 0 and prints two paths inside the environment you just built,
with the interpreter's absolute path, `ROCKE_BACKEND` and the resolved LLVM flavor
recorded. An `ImportError` here is `PYTHONPATH` or install order, not a kernel problem.

## 4. Specify the spec dataclass

Before writing kernel code, write the spec. It is the descriptor's `spec` block, and every
field of it is a field a descriptor must be able to express —
[handover-contract.md](handover-contract.md) §1.

For each field: its type, whether it has a default, and its disposition — baked, runtime
parameter, or matcher-only (§2 of the same page). Keep the no-default set small and
semantic; a missing required field is a `TypeError` at pack time, not a diagnostic.

Then, from the graph: which of these values does the graph actually vary, and which is a
property of the operation? Anything the graph varies and the spec bakes multiplies the
descriptor count by its cardinality.

**Gate:** a written field table — name, type, default-or-required, disposition — covering
every field, with no field left unclassified, and the resulting descriptor cardinality
stated as a number. "Mostly baked" is not a disposition.

## 5. Write the builder

Signature exactly `(spec, *, arch)`, one positional parameter annotated with the spec
dataclass, `arch` keyword-only, nothing else — [SKILL.md](SKILL.md) states the rule and
why a defaulted extra parameter is refused rather than discouraged.

For the authoring interface itself, read [rocke-docs-map.md](rocke-docs-map.md); this page
does not teach it. From zero, the vector-add worked example at
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/onboarding.md:101-217`
is the shortest complete path.

Write the support predicate alongside the builder. It is what `hkp_pack` consults before
building (`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:294-343`)
— and it is the last time anything will consult it, which is step 9's subject.

**Gate:** a builder module that imports under the step-3 interpreter and returns a
`KernelDef` for one concrete spec, with the returned kernel's name printed. An import that
succeeds while the builder has never been called does not discharge this.

## 6. Introspect the builder

This is the cheapest real gate in the workflow and it needs no device.

```bash
cd "$GEN"
PYTHONPATH="$ROCKE/library:$ROCKE/platform/python" "$PY" -c "
from codegen.sources import introspect
info = introspect('kernels/<arch>/<module>.py', '<builder>', spec_values={...})
print('signature_error:', info.signature_error or 'OK')
print('spec_class:', info.spec_class)
print('required:', [f.name for f in info.required_fields])
print('probed arches:', info.supported_arches)
"
```

`introspect` reports the spec class, every field, the signature verdict and the arches the
module's `supports_*` predicate accepted
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:213-221`). Pass the
config's own `spec` values; a synthesized spec usually just trips the spec's own
validation.

**On a non-empty `signature_error`:** the message names the offending parameters. Fold them
into the spec dataclass or drop them; do not add a default to make the error go away,
because a default is exactly what the check refuses. Return to step 5.

**On an empty `supported_arches`:** that means *could not be determined*, never *none*
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:84-87`). Call the
module's predicate directly with your spec and your arch, and record the answer as probed.

**Gate:** `signature_error` empty, the reported field set identical to the table written in
step 4 — every name, every required flag — and the target arch either in `supported_arches`
or accepted by a direct predicate call you ran. A field table that disagrees with
introspection means step 4 is wrong, not introspection.

## 7. Prove the numbers

Run the kernel on a device of the target architecture against the independent reference
named in step 1. rocKE's own testing guide owns the mechanics
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/testing.md:1-3`);
the launch path is at
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/runtime/compile_launch_and_manifest.md:1-3`.

Three assertions, because each corresponding failure is otherwise indistinguishable from
success:

1. The reference actually executed — not declined, not skipped.
2. The kernel actually launched and wrote its output. Fill the output buffer with a
   sentinel before launch and confirm it changed.
3. The comparison covered every output, including ones you did not expect to be
   interesting.

Run the shape set you claim, not one point of it, plus the boundaries the spec implies: the
smallest and largest shapes in the envelope, a non-tile-multiple size, and a dimension of 1.

**On a mismatch:** print the IR and the LLVM before changing anything — the debugging
patterns in rocKE's testing guide, and the failure catalog at
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/troubleshooting.md:1-6`
for the cases that are the build or the engine rather than your kernel. A `ComgrError` on
`COMPILE_SOURCE_TO_BC` is a lowering problem, not a numerics problem
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/runtime/comgr_and_hipmodule.md:208-209`).
Do not widen the tolerance to make a mismatch pass.

**If no device of the target architecture is reachable:** this gate is BLOCKED, and the
kernel is unproven. Lowering it in step 8 is still possible and still proves nothing about
numerics. Say so; do not proceed to step 9 reporting a kernel.

**Gate:** per-output pass at a stated tolerance whose provenance you state, on named
shapes, on a named device of the target architecture, with the three assertions satisfied.
A skipped reference, an unlaunched kernel or an all-sentinel comparison is a failed gate.

## 8. Lower and pack

Author the descriptor and run the packer for real. The generator config is the supported
route: `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_dense.yaml:1-27`
is a real, git-tracked worked example for a shipped rocKE kernel, showing
`kernel_source_kind: rocke` and the per-kernel split between `spec`, `metadata` and
`specialization`. The generator's own usage is at
`projects/hipdnn/tools/IngestorGenerator/README.md:37-61`.

For the descriptor tree shape, copy
`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE/gfx942_tiled_attention/`
— chosen deliberately because its builder has no refused-parameter trap
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:74-80`).

`source` is a dotted module path, not a file path
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:48-52`).
Getting this wrong is the most common first failure.

Run `hkp_pack` with the three dependencies [build-and-environment.md](build-and-environment.md)
names, and `HKP_PACK_JOBS=1` while anything is failing.

**On `unable to import rocm_kpack`:** read the *inner* module name in the message; it is
usually `msgpack` or `zstandard` missing from the interpreter, not `rocm_kpack` itself.

**On a `TypeError` hydrating the spec:** a required field is absent from the descriptor.
Compare against step 6's `required` list.

**Gate:** a `.kpack` on disk, and the *shipped* descriptor inspected to show
`kernel_source.kind == "kpack"`, a `sha256` matching the archive payload, and
`provenance.origin_kind == "rocke"` with the authored spec preserved under
`provenance.spec`
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/pipeline.py:982-1011`).
A packer exit code of 0 without reading the produced document does not discharge this.

## 9. Report and hand off

Write the completion report of [SKILL.md](SKILL.md), and the handover statement of
[handover-contract.md](handover-contract.md) §4 — the field dispositions, the probed arch
list, every baked constant the kernarg signature does not take, the launch geometry with
its deciding fields, the kernarg order and whether the ABI is conditional, and **which
pack-time checks are now unenforced together with the matcher criteria owed in their
place.**

That last item is the one nothing downstream can recover. The support predicate you wrote
in step 5 ran exactly once, at pack time, against the descriptor's spec values. Every check
in it whose inputs include a field that is not baked is now unenforced, and only you know
which those are.

State explicitly what the work does not prove: that hipDNN dispatches the kernel. No
rocKE-specific native pack exists in this tree
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:81-88`),
so a validating, loadable `.kpack` is a compiled artifact, not an integration. The native
pack, the descriptors and matchers, registration, and graphs that assert which engine
served them belong to
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md).

**Gate:** a report naming the interpreter, the backend and LLVM flavor, the device, the
shapes, the tolerance and its provenance, the `.kpack` path, and each of the six handover
facts, plus an explicit does-not-prove list. A report that omits the unenforced-check list
has not completed steps 4 through 9; it has completed steps 4 through 8.

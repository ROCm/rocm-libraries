# Kernel mining: applicability, specialization and launch contracts

Use this reference during [RUNBOOK.md](RUNBOOK.md)'s contract and native stages.
It does not choose a shipping set or own another execution sequence. The current
kernel library's authoring, dispatch, build and testing documentation and source
outrank historical examples here. For direct-load HIP, extract the same semantic
and launch facts from HIP source; do not invent a rocKE builder/profile.

## Source evidence to retain

Locate actual builder functions, their annotated dataclass types, validation and
support predicates, dispatch factories, signature/geometry helpers and launch
wrappers. Names are not conventions to guess: validation may live outside
`__post_init__`, geometry inside the builder, and ABI declarations in a
`SignatureBuilder` chain. Follow the helpers the builder actually uses.

| Source | Required observation |
|---|---|
| Builder signature and spec type | Packable `(spec, *, arch)` interface, required fields and real defaults |
| Constructor/validators/support predicate | Rejection conditions and architecture restrictions |
| Builder memory arithmetic | Baked strides, buffer bounds, loop trip counts and layout per operand |
| Dispatch factory | Request fields, constants and derived choices forming the baseline |
| Geometry and signature helpers | Grid/block formulas, deciding fields, ordered ABI slots and guards |
| Launch wrapper | Tensor shape/feature checks absent from the constructor |
| Kernel docs, history and benchmarks | Measured knob hypotheses, limitations, numeric/reference contract |

An empty introspection architecture list means unknown, not unsupported. A
constructor-valid spec is not necessarily supported; check the **final** overridden
or promoted spec with the real support predicate, including isolation and shipping
crosses. An API lookup/binding/invocation error is not a valid false predicate.
Missing/noncallable APIs, exceptions and invalid return values fail the parity or
reference tool with `ParityError`/exit 2, even with narrowing/escape flags.

Read the graph contract first. The matcher translates graph semantics into kernel
requirements; reading only Python misses graph-only optional fields and fused edges.

## Applicability classification

Classify every restriction, including constants implied by memory arithmetic:

- **Graph-only:** reject in `graph_match` if no candidate can serve it.
- **Graph versus baked value:** compare against this candidate in `kernel_match`.
  Equality is correct for a baked extent/trip count; a genuine capacity uses its
  proven bound instead. A baked buffer limit can silently zero-fill/truncate rather
  than fault, so absence of an exception is not support.
- **Knob selection:** compare the graph with each candidate's knob. For example,
  `seqlen_kv % block_n == 0` calls for a compatible candidate, not a fixed
  engine-wide sequence rejection.
- **Spec-internal:** constraints between tuning fields belong in spec construction
  and final-spec support checking, not a graph matcher.
- **Excluded semantic feature:** enforce the agreed feature exclusion. Do not imply
  that its sub-rules are implemented. Excluding requested scope requires approval.
- **Unrepresentable semantic feature:** first consider legal fused subgraphs and
  alternate field spellings. A zero search for the Python name is insufficient.
  If genuinely unavailable, report the schema limitation rather than silently
  enabling it or inventing a graph field.

A tuning field need not be compared with a graph, but it may still be needed by
scoring, geometry or workspace sizing. Include every downstream metadata consumer
in the field audit. `graph_match` runs before a candidate exists; returning
`nullopt` empties the whole engine catalog for that graph.

## Authoring intent versus observed compilation

The [generator reference](../../../IngestorGenerator/README.md#specialization-agreement)
owns the emitted specialization contract. Existing authoring profiles describe
request/dispatcher bindings, vocabulary and each KMD field's specialization or
matcher-only classification. Generation emits those declarations inside UKD
`provenance.specialization_contract`; packaging takes no separate profile list or
root manifest.

For a metadata-backed field, bind exactly one direct `field` or zero-argument
`method` on the **actual hydrated spec passed to the builder**. A direct attribute
is authoritative only if the builder consumes it without more resolution. If the
builder consumes an effective accessor, bind that accessor even when the raw field
is non-null. Retain source use sites proving the correspondence, not a copied
policy formula or a guessed `resolved_<name>` spelling.

The gfx942 dense builder illustrates why: `resolved_use_exp2_fast()` resolves the
policy intent; `resolved_use_cfvst()`, `resolved_v_row_pad()` and
`resolved_use_v_swizzle()` are the coupled decisions the builder consumes.
`use_v_swizzle=true` can still resolve false with the conflict-free-V path disabled.
Reading the raw field, or checking the accessor only when raw is `None`, misses it.

Preserve omitted/`None` policy intent in authored `provenance.spec`; do not rewrite
it to false or materialize guessed defaults. Constructor defaults/default factories
and effective accessors are observed at compilation. Unresolved `None`, missing
readouts, unsupported types, exceptions or non-repeatable resolution block full
agreement. A field with no authoritative readout is unsupported, not an automatic
matcher-only exemption. In particular, reclassifying causal or swizzle merely to
make a check pass is invalid.

Every KMD field belongs to the exhaustive disjoint specialization/matcher-only
partition. Each exemption needs a source-grounded reason, reviewed independently
of that partition's mechanical completeness. This ledger is migration/review
evidence, not another executable policy table. Metadata is completed and typed via
the referenced KMD: BOOL remains boolean, INT may deliberately project a builder
boolean to 0/1, and FLOAT is canonicalized numerically.

Compiler-owned `provenance.effective_spec` binds observed decisions and producer
origins to the actual descriptor/schema/metadata/architecture and payload. Authored
inputs cannot forge or overwrite it. Serial, prewarm and shared compile paths must
check **every** consuming descriptor independently. Full verification reads that
artifact-bound record; it does not reconstruct a spec using whichever rocKE is
installed on the checking machine. These observations establish agreement with
builder decisions, not formal equivalence of arbitrary machine code or correctness
of native dispatch.

## Layout, geometry and pointer ownership

Derive each operand's address formula independently. For example,
`((b*S+s)*H+h)*D+d` is token-major BSHD memory even if logical tensor dimensions
are listed as `[B,H,S,D]`. A kernel with no stride arguments cannot honor arbitrary
strides. Check every stride affecting an address, while allowing arbitrary strides
on extent-one axes whose index is always zero. Do not assume V shares Q's head
size or output layout without source evidence.

Record all grid/block branches, resolved constants and KMD fields they consume.
Persistent and nonpersistent launch modes can need different grids; a wrong grid
can leave output unwritten while returning success. Output shape/layout checks
belong where the graph contract guarantees those fields are available; defer to
`prepare()` when matching sees incomplete inferred output information.

For every pointer slot, name the graph UID, workspace or synthesized buffer that
supplies it, its lifetime, and its preconditions. A synthesized sequence-length
array from dims assumes uniform lengths; enforce that assumption rather than
pretending to support variable lengths. An assumption the graph cannot establish
blocks that path. Keep prepared dispatch independent of transient matching data.

## Fixed and conditional ABI are different contracts

Read the actual signature declaration in order, recording type, width, source and
presence condition for **every** slot. Do not treat all kernels as attention-dense:

- A conditional signature appends slots only under specified compile-time guards.
  C++ must replay those guards and ordering exactly.
- A fixed signature contains every slot even when a feature is disabled; that
  flag controls reads, not slot existence. Omitting an unused slot shifts all
  following arguments and corrupts memory.

The same optional features can use opposite ABI conventions in different builders.
A flat argument list is correct for a fixed ABI and wrong for a genuinely
conditional one. Source audit, launch-surface declarations and numerical device
cases all matter; metadata agreement does not verify this C++ restatement.

## Baseline and tuning evidence

Use the real dispatcher factory for baseline configs; preserve its graph-derived,
constant and effective-policy choices. A knob's legal values are not evidence that
all belong in a shipping cross-product. Kernel history and benchmarks provide
hypotheses, not a requirement to ship every explored value. Match their shapes and
numeric preconditions against the owners' published data and external workloads.

RUNBOOK measures only after a runnable baseline exists: isolation, supported
pairwise survivors, explicit selection, then regeneration/rebuild/revalidation of
the shipping artifact. A policy omission and an explicit false value may produce
different binaries; equality of displayed metadata or counts does not prove
byte-identical controls.

The contract evidence handed to the native stage includes restriction dispositions,
per-operand layout equations, graph mappings, geometry/workspace deciding fields,
ABI slot inventory, specialization binding/exemption use sites and unresolved
questions. Every unresolved correctness-relevant row blocks that path; a table's
existence or an arbitrary research time limit does not discharge it.

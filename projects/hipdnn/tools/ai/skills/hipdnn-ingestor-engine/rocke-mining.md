# Kernel contracts: applicability, specialization and launch

Current builder/dispatcher source and library documentation outrank historical
examples. For direct-load HIP, extract the same semantic/launch facts from HIP;
do not invent a rocKE builder or profile. [RUNBOOK.md](RUNBOOK.md) owns execution.

## Source evidence

| Source | Required facts |
|---|---|
| Builder and annotated dataclass | `(spec, *, arch)` interface, required fields and defaults |
| Constructor, validators, support predicate | Rejections and architecture restrictions |
| Memory arithmetic | Baked strides, bounds, loop counts and operand layouts |
| Dispatch factory | Request fields, constants and derived baseline choices |
| Geometry/signature helpers | Grid/block formulas, deciding fields, ordered ABI and guards |
| Launch wrapper | Shape/feature checks absent from construction |
| Documentation, history and benchmarks | Measured knob hypotheses and numerical limits |

Follow helpers actually consumed by the builder; do not guess naming conventions.
An empty introspection arch list means unknown, not unsupported. Constructor-valid
does not imply supported: the real predicate must accept each **final** overridden
or promoted spec, including isolation and shipping crosses. API failures are not
shape declines. Read [graph-contract.md](graph-contract.md) for graph-only features
and fused edges absent from the kernel's Python surface.

## Applicability classification

- **Graph-only:** reject in `graph_match` when no candidate can serve it.
- **Graph versus baked value:** compare in `kernel_match`, using equality for an
  extent/trip count and a proven inequality for capacity. Silent zero-fill/truncation
  is not support.
- **Knob-dependent:** test against each candidate, e.g. sequence divisibility by its
  tile, rather than imposing a fixed engine-wide constraint.
- **Spec-internal:** enforce relations between tuning fields in final-spec support
  checks, not a graph matcher.
- **Excluded semantic feature:** enforce the approved exclusion; do not imply its
  sub-rules are implemented.
- **Unrepresentable feature:** investigate semantic equivalents/composition, then
  report the missing mechanism and scope decision rather than silently enabling it.

Metadata used by scoring, geometry or workspace is still required even without a
graph comparison. Enumerate every downstream consumer.

## Authored intent and compiler evidence

The [generator reference](../../../IngestorGenerator/README.md#specialization-agreement)
owns the declaration format. The generator carries
`provenance.specialization_contract` **once on each enclosing KDP**. Inline UKDs
inherit it only when they have no own declaration; a per-UKD declaration overrides
**wholesale**, never merges. Standalone UKDs carry their own. This rule is identical
for authored and packed trees; packaging takes no separate profile list/root manifest.

Every KMD field belongs to the exhaustive, disjoint specialization/matcher-only
partition. Each specialization binding names exactly one direct `field` or
zero-argument `method` on the actual hydrated object passed to the builder. A direct
field is authoritative only if the builder consumes it without more resolution;
bind an effective accessor even when its raw field is non-null. Retain builder use
sites for each binding/exemption, independently reviewed; a mechanically complete
partition cannot justify relabeling causal or swizzle as matcher-only.

For gfx942 dense, the builder consumes `resolved_use_exp2_fast()`,
`resolved_use_cfvst()`, `resolved_v_row_pad()` and `resolved_use_v_swizzle()`.
Raw swizzle=true can still resolve false with cfvst disabled. Do not copy policy
formulas or guess a `resolved_<name>` accessor.

Keep authored omission/`None` intent in `provenance.spec`. Compilation observes real
constructor defaults/default factories and effective accessors. Unresolved `None`,
missing readouts, unsupported types, exceptions or non-repeatable resolution block
agreement; they are not wildcards. KMD completion preserves BOOL, deliberately
projects an observed builder boolean to 0/1 for INT, and canonicalizes FLOAT.

Compiler-owned `provenance.effective_spec` stays **per-kernel and is never inherited**.
It binds observations/producer origins to current descriptor/schema/metadata/arch
and payload; authored input cannot supply or overwrite it. Serial, prewarm and
shared compile results must check every consumer independently. Full verification
reads that record without importing today's producer. A packed no-specialized-field
kernel without a record is `NOT VERIFIED HERE`, not binary-agreement evidence.
These checks do not prove native dispatch or arbitrary machine-code correctness.

## Layout, geometry and ABI

Derive operand address formulas independently. `((b*S+s)*H+h)*D+d` describes token-major
BSHD memory even with logical dims `[B,H,S,D]`. A kernel without stride arguments
cannot honor arbitrary strides; extent-one axes are the exception. Do not infer
V/output width or layout from Q.

Record every grid/block branch and workspace formula, resolved constants and deciding
KMD fields. Wrong geometry can leave output unwritten without raising an error.
Output checks belong at matching when available, otherwise at preparation.

Every pointer slot needs a graph UID, workspace or synthesized-buffer source, lifetime
and enforced assumptions. Synthesizing lengths from dims assumes uniform lengths;
it is not varlen support. For a **conditional ABI**, C++ must replay presence guards
and exact order/types. For a **fixed ABI**, disabled features leave their slots in
place; omitting an unused argument shifts everything after it.

The dispatcher defines the provisional baseline. Legal knob values and historical
benchmarks are hypotheses, not shipping evidence. Policy omission versus explicit
false can change binaries even when displayed metadata/counts agree. Retain restriction
dispositions, layouts, graph mappings, geometry/workspace, ABI and binding use sites;
unresolved correctness questions block the affected path.

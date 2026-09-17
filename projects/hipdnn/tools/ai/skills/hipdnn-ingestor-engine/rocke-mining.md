# Mining a rocKE kernel for what the matcher must enforce

**You were sent here from RUNBOOK step 2b.** You owe `mining.md` on disk, every
constraint row given a verdict, before you go to step 3. This file tells you what to
extract and how to classify it.

## Read the kernel library's OWN documentation first

This page tells you what an *integrator* must extract. It is not, and must not become,
a description of how the kernel library works — that belongs to the library, which
changes it without telling this file. Where the two disagree, **the library is right
and this page is stale**.

Read these before mining, and prefer them over any restatement here:

| Source | Owns |
|---|---|
| `$PROVIDER/rocke/library/dispatch/AGENTS.md` | what the dispatcher decides and, importantly, what it does NOT — including which knobs are dispatch-invisible and where they actually live |
| `$PROVIDER/rocke/AGENTS.md`, `KERNEL_AUTHORING.md` | how kernels and specs are authored; the vocabulary the rest of this page assumes |
| `$PROVIDER/rocke/TESTING.md` | what the library's own gates do and do not prove — read its known-gaps section before trusting a green upstream result |
| `$PROVIDER/rocke/library/benchmarks/**` | what the kernel owners measure, tune and validate |
| `$PROVIDER/rocke/BUILDING.md` | building and running the library itself |

For graph corpora and workload provenance the source is the `ROCm/dnn-benchmarking`
repository and its own `docs/`; `workloads.md` in this skill covers only how an
*integration* consumes them, not how that tool works.

If something here contradicts one of those, fix this page.

**Variables and the budget, restated here because this is the file you will be inside
when they bind.** `$REPO`, `$PROVIDER`, `$GEN`, `$MODULE`, `$ARCH` are set in
`RUNBOOK.md`'s "Set these once" block — set them before running anything below.
`$M` is `$PROVIDER/rocke/library/$MODULE`, the kernel module itself. `$KDP` is the path
to your shipped `.kdp.json`, which does not exist until step 4 generates it.

**The budget: draft `mining.md` after the kernel module and its spec, before ANY third
source. Then five sources *beyond the kernel module*, maximum** — count them and name
them in the file. This document is long and every section below points at more places to
look; the cap is what stops that from becoming the whole run. Rows you are unsure of go
in marked `UNSURE` and become step-3 questions. Hitting the cap is not failure, it is the
step working.

**Why this comes after the graph contract.** Step 2a (`graph_contract.md`) already told
you what the graph can ask for, and which of the kernel's fields are derivations rather
than a direct match. Mining does not re-derive that mapping — it tells you what the
*kernel* additionally pins that the graph side never sees: spec-internal knobs, launch
geometry, ABI shape, and hard-fault conditions that live only in Python. Read 2a first;
arriving here without it means reverse-engineering the graph side under time pressure
while writing C++, which is where this skill's worst defects have come from.

A rocKE kernel's restrictions live **only** in Python. Nothing carries them into hipDNN,
and nothing checks that you got them right. An integration that skips this step advertises
a kernel that then faults or returns wrong numbers — and every mechanical check stays
green while it does.

This is a required step, not background reading. Its output is the applicability logic in
`graph_match` / `kernel_match`, plus the launch geometry and ABI in `prepare` / `launch`.

---

## What to read, and what each file gives you

| Read | Extract |
|---|---|
| `<Spec>.__post_init__` | The constraint table. Usually the single densest source. |
| `supports_<op>(spec, *, arch)` | The arch gate, and (via spec reconstruction) a re-run of `__post_init__` |
| `build_<op>(spec, *, arch)` | Address arithmetic → **the memory layout**. The most dangerous thing to get wrong. |
| `<op>_grid` / `<op>_block` | Launch geometry formulas |
| `<op>_signature` | Kernel ABI, including conditional arguments |
| module docstring + field comments | Hard-fault conditions, measured tuning facts, "not yet implemented" notes |
| `run_<op>_torch` (if present) | Launch-time guards that are NOT in the spec — often the paged/sinks shape checks |
| **`dispatch/<family>/<arch>.py`** | **The PRODUCTION configuration: which value of every tuning field the library itself ships.** See below — this is the only source that answers "what should I compile?", and the one most often skipped. |
| **`git log -- <kernel module>`** | **The knob verdicts, with their magnitudes.** Each `perf` commit is a measured A/B; a conservative guard whose message names the win it forgoes is a variant AXIS, not a setting. See *The commit history* below. |
| **`library/benchmarks/<arch>/<family>/**`** | **What the kernel team themselves measure, tune and validate.** The bench for an op is usually self-described as the NUMERIC + PERF gate for that arch — numeric correctness lives there, not in a CI pytest — so any perf claim your integration makes should be reconcilable with it. Its CLI enumerates the knobs the team consider worth exposing, and it resolves specs *through the dispatch factory*, so it measures exactly what ships. Its sweep is the shape list they care about. See below. |

**None of those names are guaranteed — locate them, do not assume them.** Roughly 40% of
the kernel modules in this tree have no `__post_init__` at all (they validate through a
free function called from the builder, e.g. `validate_common_spec` in
`kernels/common/fmha_*.py`), and roughly 45% have no named `<op>_grid`/`<op>_block`
function (the geometry is implicit in `b.block_id_*()` calls and stated only in the
module docstring). A discovery command that silently returns nothing is the failure mode
to watch for here: empty output means "wrong name", not "no rules".

**Every grep below uses `-E`.** Not decoration: `\|` alternation is a GNU-BRE extension
that several `grep` builds in this environment do not implement, and on those builds the
BRE form matches *zero* lines and exits 1 — indistinguishable from "this kernel has no
validation". Both forms were run against the real gfx942 and gfx950 dense modules: the
BRE spellings returned 0 hits on both, the `-E` spellings returned 4 and 7. If you write
one of these by hand, write it with `-E`.

```bash
# validation: whichever shape this kernel uses
grep -nE "__post_init__|^def is_valid_spec|^def supports_|^def build_" $M
grep -nE "raise ValueError|ok, why =|is_valid_spec\(|validate_.*spec\(" $M | head

# geometry, step 1: the named-function convention (about half the kernels)
grep -nA12 -E "^def [a-zA-Z_0-9]+_(grid|block|signature)\(" $M

# geometry, step 2 — run this whenever step 1 printed NOTHING. The grid is then
# implicit in the block-id reads, and the module docstring usually states it.
grep -nE "b\.(block|thread)_id_[xyz]" $M
sed -n '1,25p' $M
```

Introspect first, to get the field inventory mechanically. Use a builder name from the
`^def build_` enumeration above — there is no naming convention, so `<builder>` below is
a name you read out of the module, never one you construct:

```
python3 -c "
from codegen.sources import introspect
i = introspect('kernels/<arch>/<mod>.py', '<builder>')
print(i.signature_error or 'OK'); print(i.spec_class)
print('required:', [f.name for f in i.required_fields])
print('arches:', i.supported_arches)
"
```

Then read the source. Introspection gives you *fields*; only the source gives you *rules*.

---

## The benchmark tree: what the team themselves measure

```bash
B=$PROVIDER/rocke/library/benchmarks
ls $B/<arch>/<family>/                       # the ops they gate on this arch
sed -n '1,40p' $B/<arch>/<family>/<mode>/benchmark_*_live.py   # the header states the gate
grep -nE "add_argument|for .* in \(|range\(" $B/<arch>/<family>/<mode>/benchmark_*_live.py
```

Four things it gives you that nothing else does:

1. **Where perf is actually judged.** These files typically describe themselves as the
   arch's NUMERIC + PERF gate, with numeric correctness living there rather than in a CI
   pytest. If your integration's number cannot be reconciled with theirs, one of you is
   measuring something else.
2. **Which knobs are worth exposing.** The CLI arguments ARE the team's answer to "which
   knobs belong in the UED and the KMD" — and critically, the bench resolves specs
   *through the dispatch factory*, often with a raise-on-drift guard, so it measures what
   ships rather than a hand-built config.
3. **Which shapes are worth shipping.** The sweep loops enumerate the configurations they
   care about. Compare them against your variant set; a shape in their sweep that you do
   not cover is a gap to close or justify.
4. **The validation contract, including its holes.** Reading one is how a zero-fill vs
   NaN-sentinel discrepancy surfaced, which explained how a green upstream dashboard and
   failing hipDNN bundles were both honest (see RUNBOOK 8c-2).

**Reconcile three sources rather than trusting any one.** Each answers a different
question and no two are substitutes:

| Source | Answers |
|---|---|
| `library/benchmarks/**` | what the kernel team tune, measure and validate |
| the published results CSV (`workloads.md`, and `$GEN/tools/mine_shapes.py --published`) | which shapes they will escalate a regression on, already resolved, with priority |
| `dnn-benchmarking` servable graphs | what real callers actually ask for |
| the `supports_*` predicate | what is legal to build |

The CSV is strictly better than reading the bench source *for the shape list* — it is
already resolved, it carries priority and ticket group, and it enumerates shapes the
nested loops do not. The bench source is still worth reading for the other three items
above, which the CSV does not carry. Ask for the CSV first; read the bench anyway.

This applies to **any** rocKE op being integrated, not one family.

## The dispatcher: what to COMPILE, not what is legal

Every source above answers *"what is **legal**?"* — the validators, the layout, the ABI,
the fault conditions. **None of them answers "what value should I ship?"** That question
has one authoritative answer: the dispatcher already picks a configuration for every
request the library serves, and it has been tuned.

Find where it constructs the spec and **transcribe the whole call, field by field**:

```bash
D=$PROVIDER/rocke/library/dispatch/<family>/<arch>.py
grep -n "def _.*spec\|<Spec>(" $D                 # locate the factory
awk '/return <Spec>\(/,/^    \)/' $D             # the construction, verbatim
```

Classify **every** field into exactly one of three kinds, and put the table in
`mining.md`:

| Kind | Meaning | What your descriptor does |
|---|---|---|
| **graph** | copied from the request (extents, dtype, mask) | comes from the matched graph |
| **constant** | a fixed tuned value (`_BLOCK_N = 64`) | pin it in every variant |
| **derived** | computed from a request property (`persistent = work >= <CU count>`) | **evaluate the rule per shape and pin the result** |

**The derived kind is the trap, and it is silent.** A field you never write takes the spec
dataclass's *default*, which is chosen for a different purpose than the dispatcher's rule
— so your kernel is compiled without a lever the library turns on in production, and
nothing warns you. In one run 62 of 145 shipped variants were compiled with a
grid-strategy lever OFF that the dispatcher's rule would have turned ON, and the kernel's
own notes marked that lever validated and "KEEP everywhere". It cost a full benchmark
round-trip to find.

Constants and named policy functions are easy to spot. Rules are not: they look like
ordinary local variables several lines above the constructor. Read the *whole* factory,
not just the call.

**GATE for this section:** every field the dispatcher sets appears in your `mining.md`
table with its kind and rule; and at step 4, every one of them appears in your
descriptor's `spec` block or is justified in writing.

Note also what the dispatcher does **not** vary. Fields it leaves at their defaults are
usually knobs with a measured verdict recorded in the kernel's docstring or field
comments — swept, decided, and not worth re-sweeping. Cross-check the two: a knob the
dispatcher fixes AND the source calls proven-negative is settled, not an axis for your
variant set.

## The commit history: where the knob verdicts actually live

The spec's field comments tell you a knob's *status*. The commits that produced that
status tell you its **magnitude, its shape-dependence, and whether the fix you are
looking at is a workaround for something an AOT variant set can do properly**. That last
one is the prize, and it is invisible from the source alone.

Scope the log to the kernel module, on your own branch. This is cheap — a mature rocKE
kernel has single-digit commits.

```bash
K=$PROVIDER/rocke/library/kernels/<arch>/<module>.py
git log --oneline -- "$K" | head -20
# then read every perf / fix body IN FULL:
for c in $(git log --format=%h --grep='^perf' --grep='^fix' -- "$K"); do
  git log -1 --format='%n=== %h %s%n%b' $c
done
```

**Read the `perf` commits first.** Each is a measured A/B with numbers, a scope, and
often an explicit statement of what it could NOT fix. Three things to extract:

| Extract | Why it matters to your variant set |
|---|---|
| **The numbers** (`-15 to -21% at Sq<=1024`, `+8-10% at Sq>=4096`) | Which knobs deserve an axis at all. A 2% knob is noise; a 20% knob is your whole perf story. |
| **The axis the verdict is keyed on** | A knob gated on `seqlen` is *shape-dependent*. Your descriptors are per-shape, so you can specialize what the JIT could not. |
| **"the proper fix is X; this lands a guard as a hot fix"** | **A work item addressed to you.** |

### The pattern to hunt for

A JIT dispatcher resolves knobs once per config, so a knob whose best value depends on a
runtime shape must be handled by a **conservative gate**: pick the value that never
regresses, forgo the win everywhere else. Correct for a JIT, and it leaves measured
performance on the floor.

**An AOT variant set has no such constraint.** You compile per shape, so you can ship
*both* values and let `kernel_match` pick on the shape field. Every conservative gate in
the kernel is a variant axis you can convert into a win.

Worked example, from the gfx942 dense prefill's own history:

> `perf(rocke): hotfix - gate bf16 D128 exp2_fast to seqlen>=4096`
> "clearly regresses short-seq dense prefill (~-15 to -21% at Sq <= 1024) while winning
> at Sq >= 4096 (+8-10%)... **The proper fix is seqlen-aware kernel selection in the
> rocKE dispatcher; that work is WIP, so this lands the guard in the kernel builder as a
> hot fix.**"

Seqlen-aware kernel selection **is what an ingestor engine does**. Reading only
`_use_exp2_fast()` shows a `seqlen < 4096` branch and no reason to question it. Reading
the commit tells you it is a placeholder with a 15-21% swing behind it, waiting for
exactly the mechanism you are building.

### Mapping a knob to a variant axis

For every tuning field, the log resolves it into one of four kinds. Only the last two
belong in the cross-product:

| Kind | Signal in the commit / comment | Variant action |
|---|---|---|
| **Settled negative** | "proven-negative", "RESOLVED — do not re-attempt", "DO-NOT-PORT" | Pin at the settled value. **Emit it explicitly.** |
| **Derived** | a policy function keyed on fields you already know | Evaluate per shape, pin the result |
| **Open** | "an OPEN occupancy axis", "a sweep knob here instead of a source edit" | **Real axis.** Ship every legal value. |
| **Conservatively gated** | a guard whose commit names the win it forgoes | **Real axis.** Ship BOTH sides of the gate. |

### Emit every tuning field, including the ones you are not sweeping

A field you omit takes the dataclass default. Fine *today*, wrong the moment the default
moves — your descriptors keep serving the old value with nothing to warn you. A rocKE
spec may say this outright; the gfx942 dense spec does:

> "a harness that omits a VALUED field freezes the config at whatever the default
> happened to be the day it was written and then silently reports a stale verdict (that
> is exactly how a real regression got mis-reported in this tree)."

The exception is a **tri-state** field defaulting to `None`, meaning "resolve through the
shipping policy". Omitting those is not merely safe, it is correct — they auto-track the
kernel. Check the default before deciding: `None` -> omit, a value -> emit.

**GATE:** every tuning field in the spec appears in your config with a comment naming its
kind, or is a documented `None` tri-state. A field in neither list is a frozen default
you have not noticed.

## The classification that matters

Every rule in `__post_init__` is one of three kinds, and only some of them belong in a
matcher:

- **Graph-derivable** — checkable from tensor dims/strides/dtype or node attributes.
  → belongs in `graph_match` (graph-only facts) or `kernel_match` (graph vs. baked
  metadata).
- **Spec-internal** — a rule about *knobs the integration itself chooses* (tile sizes,
  occupancy hints, padding, persistent-CTA counts). These are not graph properties. They
  belong to whatever code builds the spec, and putting them in a matcher makes it reject
  graphs it could serve.
- **Knob-selection** — one side is a graph fact, the other is a knob *you* choose. These
  are neither, and they are the ones that break a two-bucket sort. See below.

Build a table with a graph-derivable column filled for **every** rule. Sort each rule by
asking these in order — the first YES wins:

| Ask | If yes | Bucket |
|---|---|---|
| 1. Does the rule fire only when a feature you are **not shipping** is on? | Decline that feature wholesale in `graph_match`; do NOT encode its sub-rules | **SCOPE** |
| 2. Can a caller's graph change the answer, using only tensor dims/strides/dtype and node attributes? | Continue to 3 | — |
| 3. …and does it compare against a value a **variant bakes in**? | `kernel_match`, against `$kernel.<field>` | **GRAPH/BAKED** |
| 4. …against a fixed constant instead? | `graph_match` | **GRAPH** |
| 5. Is one side a graph fact and the other a knob **you** pick? | `kernel_match` against `$kernel.<knob>`, so each variant answers for itself | **KNOB-SEL** |
| 6. Otherwise — both sides are knobs you pick | spec construction only | **SPEC** |

Question 1 is the one most often skipped, and it produces guidance that reads backwards:
a rule guarded by `if self.<feature>:` in `__post_init__` is unreachable when you ship
that feature off. Writing it into `graph_match` rejects nothing, while telling the next
reader the feature is supported. Find them mechanically:

```bash
python3 - "$M" <<'PY'
import ast, sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)

# Do NOT hardcode __post_init__. ~40% of modules validate elsewhere -- gfx942
# attention_dense keeps 23 of its rules in `supports_attention_dense` and has no
# __post_init__ at all, so a walk pinned to that one name prints nothing and reads
# exactly like "this kernel has no guarded rules". Walk every function that
# rejects, by SHAPE: a raise, or a `return False, "..."` verdict pair.
def rejects(fn):
    for n in ast.walk(fn):
        if isinstance(n, ast.Raise):
            return True
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple):
            head = n.value.elts[0] if n.value.elts else None
            if isinstance(head, ast.Constant) and head.value is False:
                return True
    return False

found = False
for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    if not rejects(fn):
        continue
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.If) and rejects(stmt):
            found = True
            nr = sum(1 for x in ast.walk(stmt) if isinstance(x, ast.Raise))
            nv = sum(
                1
                for x in ast.walk(stmt)
                if isinstance(x, ast.Return)
                and isinstance(x.value, ast.Tuple)
                and x.value.elts
                and isinstance(x.value.elts[0], ast.Constant)
                and x.value.elts[0].value is False
            )
            print(f"{fn.name}:{stmt.lineno}: guarded by  if {ast.unparse(stmt.test)}:"
                  f"  ({nr} raise, {nv} return-False)")
if not found:
    print("NO GUARDED RULES FOUND -- suspect the walk, not the kernel. "
          "Check the module really does reject somewhere: grep -nE "
          "'raise |return False,' \"$M\"")
PY
```

Anything under a guard naming a feature you decline is SCOPE. Anything ungated is
unconditional and needs a real verdict. The script prints the owning function name, so
an unexpected name (`supports_<op>` rather than `__post_init__`) is information, not a
miss.

**Rules can also be baked WITHOUT appearing in `__post_init__`, and those are the
dangerous ones** — they are invisible to a reader auditing the validation. Two sources:

```bash
grep -n "buffer_rsrc\|num_records" $M   # buffer bounds sized from SPEC fields, not tensors
grep -nE "range\(|// *BN|// *BLOCK|_STEPS|n_[a-z]*tiles" $M   # compile-time loop trip counts
```

A buffer bound computed from spec fields means a graph exceeding it reads zero-fill
rather than faulting — silently wrong. A trip count fixed at build time means a larger
graph is silently truncated to a prefix. Both are `kernel_match` equality obligations, and
neither shows up as a `raise`.

**Over-rejecting is a real bug too.** A knob that does not fit is a reason to pick a
different knob, not to decline the graph. `graph_match` returning `nullopt` empties the
**whole engine's catalog**, so an overly narrow gate there is far more expensive than an
overly narrow `kernel_match`.

### Deriving the kernel's own mapping, once a field is a derivation

`graph_contract.md` §5 already classified which of the kernel's fields are
**derivations** — concepts both sides express, but spelled differently — and named the
hipDNN side. What it cannot give you is how the *kernel itself* computes its value,
because that exists only in the Python. Getting it backwards from a "logical" guess
produces a wrong number that every mechanical check waves through; that direction is
yours to nail down here, not to re-derive from the classification table.

**Get the exact mapping from the reference executor, not from a convention.** Stage 8
verifies you against `integration-tests/gpu-ref/kernels/<op>/`, whose predicate *defines*
the mapping you are held to. Read the kernel's own key-set or formula and the reference's
side by side, and equate them term by term rather than assuming the "obvious" direction.

*(The gfx950 dense kernel's sliding-window field is the sharp instance — it isn't even
named in the schema, only derived. The kernel keeps a key iff `ktok > q - W` (the `win_lo`
window bound in the gfx950 dense builder); the reference drops iff `skv < sq - L` (the `leftBound`
branch of `GpuRefSdpaFwd.cpp`'s `score` lambda); so `W = L + 1`. The reference's own
comment states the asymmetry — "+1 on the right bound, none on the left bound" — so guessing either
direction is a silent one-key error on every masked graph. Alignment is the same shape of
problem: the reference shifts by `offset = topLeft ? 0 : (Skv - Sq)`; a kernel masking on
raw absolute token indices implements one alignment convention only, and the other is
servable exactly when a shape restriction the reference states holds. Declining it
outright instead of checking that restriction is the over-rejection trap — every shipped
bundle exercising that alignment happens to meet the restriction, so a blanket decline
would silently serve none of them.)*

### A third kind: the knob-selection constraint

A rule that compares a graph fact against a knob you choose belongs in **neither**
bucket, and forcing it into one produces a bug either way. From the gfx950 dense kernel:

```python
seqlen_kv % block_n == 0
sliding_window % block_n == 0
```

`seqlen_kv` and `sliding_window` come from the graph. `block_n` is a tuning knob the
integration picks. Sort this "graph-derivable" and the matcher declines graphs a
different `block_n` would serve; sort it "spec-internal" and the matcher admits graphs
the shipped variant faults on.

**The correct handling is selection, not rejection.** The matcher's job is to pick a
variant whose `block_n` divides this graph's `seqlen_kv`, and to decline only when *no*
shipped variant does. Concretely: `kernel_match` tests `seqlen_kv % $kernel.block_n == 0`
against the candidate kernel rather than a fixed constant, so each variant answers for
itself and `score` ranks whichever survive.

Mark these rows `KNOB-SEL` in the table with both operands named. They are also a
variant-set input: the set of `block_n` values you ship determines which sequence lengths
the engine can serve at all.

### Two more buckets you will hit


**Unrepresentable — a real capability with no hipDNN attribute to carry it.** Some rocKE
features change *semantics* (so they are not tuning knobs) yet have no corresponding
field anywhere in the hipDNN graph schema (so they are not graph-derivable either). Grep
the schema tree for the spec field's name; a real hit count of zero is the signature.
*(Two boolean mode flags on a gfx950 tiled-attention kernel are this today — grep
`projects/hipdnn` for either name and nothing comes back.)*

Do not force these into the other two buckets. The honest handling is:

- Never author a UKD variant that turns the feature on — no graph can ever request it, so
  the kernel would be built for a mode the frontend cannot express.
- Fix the spec field to its off value and say so in the descriptor's provenance.
- Flag it in your Step 9 report as a **schema gap**, not an integration gap. Closing it
  needs an RFC-level change to the graph attributes, which is outside this skill.

**Bound-checked rather than equality-checked.** A graph fact is usually compared against a
baked value for *equality* — but sometimes the baked value is a **capacity**, and the
correct test is an inequality: the kernel serves any runtime value up to what it was
compiled for, not only the one value a naive equality check would demand. *(A gfx950
tiled-attention kernel compiles a binary-search iteration count from its sequence-length
bound, and correctly serves any runtime sequence length the resulting bound covers; an
equality check there would decline graphs the kernel handles perfectly.)*

When a KMD field is derived from a graph quantity rather than equal to it, say so in the
table and write the inequality into `kernel_match`. Read the field's own derivation in the
Python to find the direction of the bound.

---

## Layout: read the address arithmetic

The kernel's memory layout is not declared anywhere. Derive it from `build_<op>`:

```python
stride_q_tok = Hq * D                       # elements between consecutive query tokens
q_base = bt * Sq * stride_q_tok + hq * D
addr   = q_base + q_tok * stride_q_tok + col
```

That is `((b*S + s)*H + h)*D + d` — row-major `[B, S, H, D]`, i.e. **BSHD**, head varying
faster than sequence. Note what is *absent*: no stride arguments. The kernel computes
strides from `H` and `D`, so it cannot accept a permuted or sliced tensor at all.

Check every operand separately: in that kernel V reuses K's base and stride (so V must
share K's exact layout), and O reuses Q's (so O must match Q exactly).

**Consequence to write into the matcher:** a graph in another layout is read as if it were
this one — in-bounds, wrong elements, **no fault**. Bounds-checked buffer loads on the K/V
side turn OOB into zero-fill, which is also silent. Only an undersized Q on the unguarded
path actually faults.

---

## Launch geometry and ABI

Restate both in C++, citing the Python line.

**Geometry** — mind the branches:

```python
def <op>_grid(spec):
    if spec.persistent: return (spec.num_persistent, 1, 1)
    nqb = (spec.seqlen_q + _BLOCK_M - 1) // _BLOCK_M
    return (nqb, spec.num_query_heads, spec.batch)

def <op>_block(spec):
    return (spec.num_waves * 64, 1, 1)      # num_waves = _BLOCK_M // 32
```

Resolve the constants (`_BLOCK_M = 256` → block `(512,1,1)`) rather than carrying symbols
you cannot see from C++.

**ABI** — read `<op>_signature`, or the builder's `b.param(...)` declarations in order,
and **verify the shape per kernel. Do not assume a template.** rocKE kernels differ
fundamentally here, and getting it wrong corrupts memory rather than erroring:

- **Some kernels append optional arguments conditionally** — the argument *count* depends
  on which optional features the spec turns on, and `launch()` must replay the same
  conditionals to stay aligned. *(A dense attention kernel's signature growing by
  `[sink_ptr]` when sinks are enabled, then further by variable-length and paged-specific
  pointers under their own flags, is the instance worth knowing: three independent
  conditionals stacking in the same signature.)*
- **Others declare every slot unconditionally** — every optional pointer is a plain
  `b.param(...)` with no `if` around it, and a compile-time flag decides only whether the
  kernel *reads* the slot, never whether the slot *exists*. Here the argument count is
  fixed, and a "conditional" `launch()` would misalign every pointer after the first
  disabled feature. *(A tiled attention variant that declares its optional pointers this
  way is the instance: same three-feature shape as above, opposite ABI convention.)*

Determine which by reading the `b.param` declarations line by line and noting whether
each sits inside an `if`. Then mirror exactly that in `launch()`, and cite the Python
lines in a comment — nothing checks this correspondence.

### Every pointer needs a named source. Synthesis is an applicability rule.

Getting the ABI *aligned* is not the same as getting it *right*. For each pointer
parameter, write down which of these it is — the table is the deliverable, one row per
pointer, and a pointer with no row is an unfinished audit:

| Source | Meaning |
|---|---|
| a graph tensor | the UID whose device buffer you pass |
| a value the graph supplies | scalar or metadata, named |
| **synthesised in `launch()`** | you are computing it — see below |

**A synthesised pointer is a claim about the inputs, and the claim belongs in
`graph_match`.** A shipped integration filled `query_start_len_ptr` and `seq_lens_ptr`
by computing offsets from tensor dimensions — correct only when every sequence in the
batch has the same length, and silently wrong otherwise. Nothing failed: the kernel
launched, the graph ran, the numbers were wrong for non-uniform batches only, and no
gate covers arithmetic nobody declared.

So: if you synthesise a pointer, state the assumption that makes the synthesis valid,
and **enforce that assumption in `graph_match`** — decline the graphs it does not hold
for. An assumption you cannot enforce is a decline, not a default. The enumeration is
mechanical: the parameter list you just read line by line *is* the checklist, so do this
while you have it open rather than at step 6 when the ABI is a memory.

---

## `supports_*` is necessary, not sufficient

Typical shape:

```python
def supports_<op>(spec, *, arch="gfxNNN"):
    if arch != "gfxNNN": return False, f"... gfxNNN-only (got {arch})"
    try: <Spec>(**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()})
    except ValueError as e: return False, str(e)
    return True, ""
```

It checks the arch string and re-runs `__post_init__`. It therefore says **"is this spec
self-consistent"**, not "does this kernel serve this graph." It knows nothing about
layout, nothing about whether the operand tensors exist or are the right shape, and
nothing about the launch-time guards in `run_<op>_torch`.

Use it as a filter after you have derived a candidate spec from the graph. Never as the
matcher's only gate.

---

## Three traps worth stating outright

**A hard-fault condition may need a different variant, not a rejection.** If the kernel
faults when `seqlen_q % 256 != 0` on the aligned path but a `ragged=True` variant handles
exactly that case, the right response is to select the ragged variant. Rejecting is safe
but wrong; ignoring is a fault.

**"Not yet implemented" guards read like capabilities.** A spec may reject
`paged` + plain-causal while accepting `paged` + sliding-window — the *narrower* case is
the supported one. Read the guards; do not infer support from the general shape of the
feature.

**A baked constant is an applicability rule even when it looks like a shape.** This is
the one that has actually shipped bugs.

### Worked example: the `batch == 1` defect

The gfx950 dense kernel's builder module — the one your `MODULE` points at — reads:

```python
if self.batch != 1:
    raise ValueError("paged multi-sequence (batch>1) not yet implemented")
```

`batch` looks like an ordinary problem dimension — the kind you assume any kernel handles.
Here it is a hard capability bound on the paged path. A real integration missed it: the
kernels were compiled for `batch == 1`, nothing in the matcher checked it, so the engine
advertised itself for multi-batch graphs it could not serve. The symptom appeared far
downstream as failures in the shared integration suite, where the cause is expensive to
find.

Two independent mistakes produced it, and both are avoidable here:

1. **The applicability gap.** A constant baked into the compiled kernel was not mirrored
   into `graph_match`/`kernel_match`. Anything a variant bakes in — batch, sequence
   length, head count, dtype, a mode flag — is a matcher obligation. If the descriptor's
   `spec` pins a value, the matcher must require the graph to match it, or the KMD must
   carry it and `kernel_match` must compare it.
2. **The variant gap.** Shipping only `batch == 1` left every other batch size unserved.
   Once applicability is honest, the narrow variant set becomes visible as *declining
   graphs* — the correct, loud failure, and the signal to widen the set.

**The check to run — and it is exhaustive, not a hunt for one sneaky field.** `batch` is
the example because it shipped, not because it is special. The rule is: **every field a
variant pins needs a matcher check** — either the KMD carries it and `kernel_match`
compares it, or `graph_match` requires it outright.

**Do not copy a field list out of this document.** Any list written here is one kernel's
answer and will be wrong for yours: the gfx950 dense bundle pins **18** spec fields, and
the gfx942 dense kernel additionally bakes `waves_per_eu` and seven gfx942-only fields
(`block_m`, `iglp`, `lds_row_pad`, `use_cfvst`, `use_exp2_fast`, `use_v_swizzle`,
`v_row_pad`) into its `kernel_name()`. Enumerate yours from your own descriptors:

```bash
python3 - "$KDP" <<'PY'
import collections, json, sys
doc = json.load(open(sys.argv[1]))
pins, meta = collections.defaultdict(set), set()
for k in doc["kernelDescriptors"]:
    spec = k.get("kernel_source", {}).get("spec") or k.get("provenance", {}).get("spec", {})
    for f, v in spec.items():
        pins[f].add(json.dumps(v))
    meta |= set(k.get("metadata", {}))
print(f"{len(pins)} spec fields pinned by this variant set:")
for f in sorted(pins):
    where = "in KMD" if f in meta else "NOT IN KMD -- needs a graph_match check"
    print(f"  {f:22} {len(pins[f])} distinct value(s)   {where}")
PY
```

Every row needs a verdict. A field with one distinct value is the dangerous case, not the
safe one: it is a constant your kernels were compiled with, invisible in the KMD, and
exactly the shape of the `batch == 1` defect above.

So enumerate mechanically rather than by eye. Walk your descriptor's `spec` block field
by field, and for each one write down the matcher check that enforces it and where it
lives. A field with no check is this defect. Do it as a table in `mining.md`:

| Spec field | Pinned to | Enforced by | Hook |
|---|---|---|---|
| `head_size` | 128 | `attrs.head_size == $kernel.head_size` | `kernel_match` |
| `batch` | 1 | `problem.batch == 1` | `kernel_match` |
| ... | ... | **MISSING** ← the defect | |

Reading the list and thinking "those look fine" is exactly how the shipped bug shipped.
Put every unenforced row in your rejection checklist at Tier 1 — silent wrong answers —
because the engine will accept work it cannot do.

### When verification fails, suspect applicability first

If a graph produces wrong numbers, faults, or fails in the shared suite, **check
applicability before you debug the kernel.** In this system the overwhelmingly likely
cause is that the matcher accepted a graph the kernel was never built for. Order of
investigation:

1. Which variant did the engine select, and what does its `spec` actually pin?
2. Does the failing graph differ from those pinned values on *any* axis — batch, layout,
   sequence length, head counts, dtype, a mode flag?
3. If yes, the bug is the matcher, not the kernel. Tighten `graph_match`/`kernel_match`
   so the graph is declined, then decide whether to add a variant that serves it.

A kernel that computes the wrong answer for a problem it was never compiled for is
behaving correctly. The defect is upstream.

---

## KMD fields are not only matcher inputs

A KMD field earns its place if **anything** downstream reads it — and matching is only one
consumer. Per RFC 0017, matchers, the heuristic, workspace sizing and dispatch all read
kernel metadata as `$kernel.<field>`, so a value your `prepare()` needs to size a grid
belongs in the KMD even when no matcher ever compares it.

Concretely, for a rocKE kernel, sweep the spec for fields in these roles:

| Role | Consumer | Example |
|---|---|---|
| Distinguishes two compiled variants | `kernel_match` | dtype, head size, causal flag |
| Feeds the launch geometry | `prepare()` | warps-per-CTA, rows-per-warp, tile size |
| Sizes scratch | `workspaceBytes` | anything the kernel's own scratch formula reads |
| Ranks candidates | `score` / UHD | the free performance knob |

RFC 0017 §7 works this exact problem using `build_unified_attention_2d_tiled` as its
example. If your kernel's grid formula reads a spec field, that field is a KMD field —
otherwise `prepare()` has to re-derive it from nothing, and the descriptor no longer
describes the kernel it names.

---

## Deliverable of this step

Before writing the pack, produce:

1. The constraint table, graph-derivable column filled for every row, with the
   unrepresentable and bound-checked cases called out.
2. The layout statement, with the arithmetic that proves it, per operand.
3. Grid/block formulas with constants resolved, **and the KMD fields they read**.
4. The ABI list, stating per argument whether its slot is conditional or always present.
5. A rejection checklist ordered by failure severity — **silent wrong answers first**,
   faults second, missed-opportunity last. That ordering is the implementation order for
   `graph_match`: the checks whose absence corrupts results go in first.

Show 1, 2 and 5 to the human when you present the batch confirmation, plus any
unrepresentable feature you found. These are exactly the decisions they can correct and
you cannot verify alone.

---

## GATE

```bash
ls mining.md
grep -c '^|' mining.md      # the constraint table has rows
```

`mining.md` exists, every row of the constraint table carries a verdict (a bucket, or
`UNSURE`), the layout statement names the arithmetic that proves it, and the ABI list
states per argument whether its slot is conditional. Rows still `UNSURE` at the source
cap are not a blocker — they are step 3's questions.

Then go to step 3 and send the batch message. You now know what the kernel can answer;
the human decides what to build.

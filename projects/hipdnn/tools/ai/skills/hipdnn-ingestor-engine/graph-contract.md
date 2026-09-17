# The graph contract — what hipDNN can ask your kernel to do

**You were sent here from RUNBOOK step 2a.** You owe `graph_contract.md` on disk before you
start step 2b (mining the kernel). This file tells you what to put in it.

**Variables.** `$REPO` and `$OPTABLE` (your op's FlatBuffers table, e.g.
`SdpaAttributes`) are set in `RUNBOOK.md`'s "Set these once" block. Set them before
running any command below — the first one uses `$REPO`.

**Why this comes first.** The matcher you write in step 6 is a translation between two
descriptions of the same operation: the kernel's, which lives in Python, and hipDNN's,
which lives in a schema. Miners who start on the kernel side arrive at the matcher having
never read the graph side, and then reverse-engineer it under time pressure while writing
C++. Every silent-wrong-answer defect this skill knows about came from that order.

The graph side is also the smaller job. It is a schema, a header and a few sample graphs —
an hour, mostly reading, entirely mechanical.

---

## Deliverable

`graph_contract.md`, with five sections, in this order:

| # | Section | Done when |
|---|---|---|
| 1 | The operation match | You name the node(s), and for a fusion the UID edges |
| 2 | The field audit | Every field of every table you match is accounted for |
| 3 | The frontend reading | Intent, defaults and deprecations noted from the header, cross-checked against the framework operator |
| 4 | A real graph | You have read an actual graph of this op from BOTH the in-tree bundles and the benchmarking workloads, and recorded where they disagree |
| 5 | The disagreement table | Every kernel field mapped to its hipDNN spelling, or marked absent |

Sections 1 and 5 are the ones step 6 consumes directly. Section 5 is the single most
valuable artifact this step produces.

---

## 1. Which operation — or composition of operations — is this?

**hipDNN is a graph API.** An operation is a *subgraph*, not necessarily a node. A kernel
that computes something no single table names is the normal case for a fused kernel, and
**not** evidence of a gap. Evaluate the kernel against the whole API — nodes *and* their
legal compositions — before concluding anything.

### The two levels of the catalog

**Nodes** — the authoritative list, one enum entry per node type:

```bash
sed -n '/^enum class NodeType/,/^};/p' \
    $REPO/projects/hipdnn/frontend/include/hipdnn_frontend/node/NodeType.hpp
ls $REPO/projects/hipdnn/flatbuffers_sdk/schemas/*_attributes.fbs
```

**Compositions** — which chains actually ship, are tested, and have reference coverage.
Read this before assuming your fusion is novel:

```bash
cd $REPO/dnn-providers/integration-tests/integration-test-bundles
python3 - <<'PY'
import json, glob, collections
seen = collections.Counter()
for f in glob.glob('**/graph.template.json', recursive=True):
    try: nodes = json.load(open(f)).get('nodes', [])
    except Exception: continue
    if len(nodes) > 1:
        seen[' -> '.join(n.get('type', '?') for n in nodes)] += 1
for chain, count in seen.most_common():
    print(f'{count:3}  {chain}')
PY
```

On the tree this was written against that prints nine distinct chains — convolution and
matmul and both batchnorm flavours each followed by pointwise, a three-node
dequantize/dequantize/matmul, and so on. **None of those chains has a table of its own.**
There is no `ConvolutionFwdPointwise` in any `.fbs`; the fusion is expressed entirely by
tensor-UID edges between existing nodes. If you grep the schema directory for your fused
kernel's name and find nothing, that tells you nothing at all.

### State one of four outcomes

Match on what the kernel **computes**, never on what it is called. rocKE and hipDNN use
different vocabularies for the same mathematics.

| Outcome | What to record |
|---|---|
| **One node matches** | The node type and its `.fbs` table. |
| **A composition matches** | Every node **and the UID edges between them** — that edge list is what `graph_match` walks, so it is a deliverable, not a note. Say whether you serve the whole subgraph or one node of it. |
| **Several readings match** | All of them, which one you serve, and what happens to the rest. A forward/backward pair, or a fusion that is also expressible as one node with an optional field set, both land here. |
| **None match** | Do not conclude this cheaply. Work the disconfirmation checklist below. |

`graph_match` receives the whole graph, so a fused kernel is directly implementable. Some
shipped packs open with `nodeCount() != 1` — that is those packs choosing to serve
single-node ops, **not** a framework limit. Do not read it as one.

---

## 2. What can the graph ask for?

Every field of **every** table your match covers — one table for a single node, all of
them for a subgraph. Each field is either **consumed** (you read it and act on it) or
**explicitly rejected**. There is no third category: an unchecked field is accepted and
then silently not honoured, which is a wrong answer with no error.

```bash
sed -n "/^table $OPTABLE/,/^}/p" $REPO/projects/hipdnn/flatbuffers_sdk/schemas/*.fbs
```

Watch for one feature spelled several ways — a mode, its tensor UID, and its scale may be
three separate fields, and rejecting only the first admits graphs carrying the rest.

**For a fusion, also account for the edges.** An intermediate tensor's dtype and layout are
part of the contract and belong to *neither* node's attribute table. A fused kernel almost
always assumes something about its intermediates; write those assumptions down here,
because nothing in the schema will state them for you.

---

## 3. How does the frontend spell it?

The `.fbs` is the wire format. The frontend header carries the **intent**:

```bash
ls $REPO/projects/hipdnn/frontend/include/hipdnn_frontend/attributes/
ls $REPO/projects/hipdnn/frontend/include/hipdnn_frontend/node/
```

Read the attribute header for your op alongside its table. It gives you setter names,
defaults, doc comments explaining what a field *means*, and — critically — deprecation
notes. **A field can be present, settable, and the wrong thing to read.** Deriving
behaviour from a deprecated field that modern callers leave at its default produces a
matcher that is green on every test and wrong on every real graph.

Where the header and the schema disagree, the loader wins. Note the disagreement anyway;
it usually means one of them is mid-migration.

**Cross-check the semantics against the framework operator for the same op.** hipDNN's
frontend is deliberately close to cuDNN's, and both are close to the operator callers
actually write against — so the framework API is a third statement of the same contract,
in a vocabulary you can check against documentation rather than infer. Find the operator
that computes what your kernel computes (`torch.nn.functional`, `torch.linalg` and
`torch.Tensor` methods cover most of the catalog; the name will not match hipDNN's) and
read its signature:

```bash
python3 -c "import torch.nn.functional as F; print([n for n in dir(F) if '<substring>' in n])"
python3 -c "import torch.nn.functional as F; help(F.<operator>)"
```

What it is good for, specifically:

- **Which fields are genuinely optional**, and what a caller omitting them means. If the
  framework op defaults a scalar and hipDNN marks it optional, that is a real default a
  graph may rely on. If the framework *requires* it, an absent value in a graph is a
  caller error rather than a value to invent. (§5 and `native-pack.md` are both emphatic
  that inventing a default for a scalar the graph did not supply is a silent wrong
  answer; the framework op is how you tell the two cases apart.)
- **Which concepts are one feature under several spellings.** A single framework
  argument frequently maps to a cluster of schema fields — one mode flag, one tensor
  uid, one scale — and rejecting only one of the cluster admits graphs carrying the rest.
- **What the op does NOT have.** A capability with no framework spelling and no schema
  field is a strong signal for the *unrepresentable* bucket in §5, not something to hunt
  for in the graph.

This is also the reference RUNBOOK 8e validates against (`--validate pytorch`), so
understanding its argument list now is what makes a numeric disagreement there
interpretable later rather than mysterious.

---

## 4. What does a real graph look like?

Read one. Not the directory name — the JSON.

**Three sources, and they disagree with each other in ways that matter.** Read at least
the first two; the third is what tells you whether your engine is useful.

```bash
# 1. In-tree samples and the committed test bundles.
ls $REPO/projects/hipdnn/samples/
ls $REPO/dnn-providers/integration-tests/integration-test-bundles/quick/

# 2. Real workloads, from the benchmarking repo. SAME graph JSON schema, so these load
#    with the same reader -- but they are traces of production models rather than cases
#    someone authored to exercise an engine.
#    (github.com/ROCm/dnn-benchmarking; RUNBOOK 8e runs against these for real.)
dvc pull Workloads/models/<model>.tar.gz.dvc Workloads/microbench/<lib>.tar.gz.dvc
tar tzf Workloads/models/<model>.tar.gz | head

# 3. The framework operator for this op -- not a graph, the SEMANTICS: what arguments it
#    takes, what it makes optional, what it defaults. Read in section 3 above; it is the
#    API your callers came from, and the reference RUNBOOK 8e validates you against.
```

**Why more than one source.** The committed bundles were authored to exercise whatever
engine shipped first; the benchmarking workloads are what callers actually send. Where
they differ, your matcher inherits whichever you happened to read — so read both and
write down the differences. Three axes are worth checking explicitly, because each one
silently decides a matcher:

- **Memory layout.** Two sources can carry the same logical dims and different strides.
  A kernel that bakes a layout is exercised by one source and declines the other
  *entirely*, and which you read first decides whether that reads as a bug or a feature.
- **Spelling of an optional feature.** A schema often carries more than one way to
  request the same thing, typically a modern field set plus a deprecated one it replaced.
  Different producers pick different spellings. A matcher that reads only the convention
  its own test bundles use passes the whole suite and mis-serves production graphs — §3
  and §5 both warn about this, and the external workloads are where you *see* it.
- **Shape magnitude.** Authored test cases are small on purpose; real traces are not, and
  the gap is often orders of magnitude. For a kernel that bakes its extents, that gap
  *is* the variant set.

**Extract the distinct shape tuples while you are here.** Sequence lengths, batch sizes,
head counts — for your op, across the model traces. That list is a direct input to
RUNBOOK § *Sizing the variant set* (item 3, workload realism): a variant set chosen
without it satisfies feature coverage and tuning headroom, passes every gate in the
runbook, and still declines every graph a real model sends.

Read this now, at step 2a, rather than discovering at 8e that the engine serves its own
test suite and nothing else.

*(Worked instance, for shape only — do not reuse the specifics: on the tree this was
written against, the in-tree SDPA bundles and the shipped model traces disagreed on all
three axes at once — opposite stride orders, opposite mask spellings, and sequence
lengths two orders of magnitude apart. Your op's three answers will be different; the
three questions are the transferable part.)*

Then dump its node/edge structure, which is exactly the shape your matcher will walk:

```bash
python3 - "<a graph.template.json or bundle .json>" <<'PY'
import json, sys
nodes = json.load(open(sys.argv[1])).get('nodes', [])
produced = {}
for i, n in enumerate(nodes):
    label = n.get('name') or n.get('type')
    ints = lambda d: {k: v for k, v in (d or {}).items() if isinstance(v, int)}
    print(f"[{i}] {label}  in={ints(n.get('inputs'))}  out={ints(n.get('outputs'))}")
    for v in ints(n.get('outputs')).values():
        produced[v] = label
for n in nodes:
    label = n.get('name') or n.get('type')
    for k, v in ((n.get('inputs')) or {}).items():
        if isinstance(v, int) and produced.get(v, label) != label:
            print(f"EDGE uid={v}: {produced[v]} --> {label} ({k})")
PY
```

This is what makes *"will a real graph actually match my matcher"* answerable now, rather
than at stage 8 on a GPU. Write the answer down.

---

## 5. Where do the kernel and hipDNN disagree?

**The deliverable step 6 consumes.** One row per field the kernel pins:

| Kernel field | hipDNN spelling | Kind | Note |
|---|---|---|---|
| … | … | same name / different name / **derivation** / **no hipDNN field** / **no rocKE field** | … |

The last three kinds are where defects live:

- **Derivation** — both sides express the concept, differently. The rule is a computation,
  not a comparison, and the computation is yours to get right. If an incumbent engine
  already serves this op, it has almost certainly solved the same derivation: look for a
  `plans/` directory under `src/engines/<incumbent>/` (not every engine has one) and read
  it before you invent your own. So is the
  reference executor under `integration-tests/gpu-ref/kernels/<op>/`, whose predicate
  *defines* the mapping you will be verified against — off-by-ones included.
  *(SDPA is the sharp example: hipDNN has no `causal` boolean at all. Causality is derived
  from bound fields, with deprecated booleans taking precedence only when set — and a
  matcher reading only the deprecated pair computes "not causal" for every modern graph.)*
- **No hipDNN field** — the kernel has a capability the graph cannot request. Never ship a
  variant with it enabled; no graph can ever select it. Fix the spec field to its off value
  and report it at stage 9 as a **schema gap**, not an integration gap.
- **No rocKE field** — the graph can ask for something the kernel knows nothing about.
  These are invisible from the Python side, so this table is the only place they surface.
  They must be **explicitly rejected** in `graph_match`.

For a fusion, add a row per edge for anything the kernel assumes about an intermediate.

---

## When nothing matches

This verdict stops the run and asks a human for schema work, so the bar is deliberately
high. It is also the verdict you will *want* to reach when an operation is unfamiliar,
because "hipDNN cannot express this" is a more comfortable conclusion than "I do not
understand this API yet." **Most first-pass no-match calls are the second thing.**

Work all six checks and record each result in `graph_contract.md`. None takes long.

1. **Semantics, not vocabulary.** Ask what the kernel computes, then find the node that
   computes it. A name absent from `NodeType.hpp` is not evidence of anything.
2. **Decompose it.** *The check most likely to overturn a no-match.* Write down the
   sequence of mathematical steps the kernel performs, then ask of each: is this a node?
   Fused epilogues, activation tails, pre-scales, dequantize-then-multiply,
   normalize-then-pointwise — all are compositions, and none has a table. If every step
   maps to a node, the operation is expressible **today** and your answer is a subgraph
   match. Check the shipped chains from section 1 to see which are already verifiable.
3. **The inverse: is it one node with fields set?** Many kernels are a *mode* of an
   existing operation rather than a new one — a masking rule, a layout, a quantization, an
   optional tensor. Read the candidate table's optional fields before rejecting its node.
4. **Is it partly expressible?** If the kernel fuses three steps and two are nodes, the
   honest answer is usually "serve the two-node subgraph, decline the rest" — a **scope
   decision**, not an escalation. A narrow engine that ships beats a wide one that blocks.
   Say what you would decline.
5. **The frontend and the cuDNN compatibility shim.** A concept absent from the schema may
   be reachable through a frontend setter or carried by the shim under a cuDNN name. Both
   record intent in comments the schema does not.
6. **Is it in flight?** `projects/hipdnn/docs/Roadmap.md` and `docs/rfcs/` cover planned
   operations. "Not yet" and "never" are different answers.

**If all six fail, escalate — and stop.** Do not route around it. Report: the kernel's
semantics, its step-by-step decomposition with the steps that had no node, the closest
existing operation and precisely why it does not fit, and which check came closest.
Adding an operation is an RFC-level change with its own process
(`projects/hipdnn/docs/AddingNewOperations.md`); it is not this skill's to make.

**`CUSTOM_OP` is not the escape hatch it appears to be.** It exists for plugin-private
operations, and **no reference executor covers it** — so a `CUSTOM_OP` integration cannot
be numerically verified and **cannot reach stage 8**, which the completion contract
requires. Choosing it converts a schema conversation into a permanently unverifiable
engine. If it is genuinely right, a human decides that explicitly, knowing the cost.

---

## GATE

```bash
ls graph_contract.md
grep -c '^|' graph_contract.md      # the disagreement table has rows
```

`graph_contract.md` exists, section 1 names the node(s) and any UID edges, section 2
accounts for every field of every matched table, section 4 records what the in-tree
bundles and the benchmarking workloads each look like **and where they disagree**, and
section 5 has a row per pinned kernel field. A no-match escalation additionally
carries all six disconfirmation results.

Then go to step 2b and mine the kernel. You now know what the graph can ask; mining tells
you what the kernel can answer.

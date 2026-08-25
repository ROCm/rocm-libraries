# Mining a rocKE kernel for what the matcher must enforce

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

Introspect first, to get the field inventory mechanically:

```
python3 -c "
from codegen.sources import introspect
i = introspect('kernels/<arch>/<mod>.py', 'build_<op>')
print(i.signature_error or 'OK'); print(i.spec_class)
print('required:', [f.name for f in i.required_fields])
print('arches:', i.supported_arches)
"
```

Then read the source. Introspection gives you *fields*; only the source gives you *rules*.

---

## The classification that matters

Every rule in `__post_init__` is one of two kinds, and only one of them belongs in a
matcher:

- **Graph-derivable** — checkable from tensor dims/strides/dtype or node attributes.
  → belongs in `graph_match` (graph-only facts) or `kernel_match` (graph vs. baked
  metadata).
- **Spec-internal** — a rule about *knobs the integration itself chooses* (tile sizes,
  occupancy hints, padding, persistent-CTA counts). These are not graph properties. They
  belong to whatever code builds the spec, and putting them in a matcher makes it reject
  graphs it could serve.

Build a table with a graph-derivable column filled for **every** rule. Example, from the
gfx950 dense-attention kernel:

| Rule | Graph-derivable? | Where it goes |
|---|---|---|
| `dtype in {bf16, fp16}` | YES — off the Q/K/V tensor | `graph_match` |
| `head_size in (64, 128)` | YES — last dim | `graph_match` |
| `num_query_heads % num_kv_heads == 0` | YES — both are dims | `graph_match` (GQA) |
| `sliding_window > 0 requires causal` | YES — node attributes | `graph_match` |
| `paged` requires `head_size == 128`, `batch == 1`, `sliding_window > 0` | YES | `graph_match` |
| `block_n % 32 == 0` | NO — a knob you pick | spec construction |
| `waves_per_eu` in `[1,8]`, `lds_k_group_pad % 8 == 0` | NO — tuning knobs | spec construction |
| `persist_decode` enum, `num_persistent > 0` | NO — launch strategy | spec construction |

**Over-rejecting is a real bug too.** A knob that does not fit is a reason to pick a
different knob, not to decline the graph.

### Two more buckets you will hit

**Unrepresentable — a real capability with no hipDNN attribute to carry it.** Some rocKE
features change *semantics* (so they are not tuning knobs) yet have no corresponding
field anywhere in the hipDNN graph schema (so they are not graph-derivable either).
`has_softcap` and `use_qq_bias` on `attention_tiled_2d` are both this: grep
`projects/hipdnn` for either and you get nothing.

Do not force these into the other two buckets. The honest handling is:

- Never author a UKD variant that turns the feature on — no graph can ever request it, so
  the kernel would be built for a mode the frontend cannot express.
- Fix the spec field to its off value and say so in the descriptor's provenance.
- Flag it in your Step 9 report as a **schema gap**, not an integration gap. Closing it
  needs an RFC-level change to the graph attributes, which is outside this skill.

**Bound-checked rather than equality-checked.** A graph fact is usually compared against a
baked value for *equality* — but sometimes the baked value is a **capacity**, and the
correct test is an inequality. `attention_tiled_2d` compiles `binary_search_iters` from
`num_seqs`, and that kernel correctly serves any runtime `num_seqs <= 2^iters - 1`. An
equality check there would decline graphs the kernel handles perfectly.

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

- **Some append optional arguments conditionally.** `attention_dense`'s signature grows
  by `[sink_ptr]` when `use_sinks`, `[cu_seqlens_q, cu_seqlens_kv]` when `varlen`,
  `[block_tables, kv_lens, block_table_stride]` when `paged` — so the argument *count*
  depends on the spec, and `launch()` must replay the same conditionals.
- **Others declare every slot unconditionally.** `attention_tiled_2d` declares
  `sink_ptr`, `alibi_slopes_ptr` and `qq_bias_ptr` as plain `b.param(...)` with no `if`
  around them; the compile-time `USE_SINKS`/`USE_ALIBI`/`USE_QQ_BIAS` flags decide whether
  the kernel *reads* the slot, not whether the slot *exists*. Here the argument count is
  fixed and a "conditional" `launch()` would misalign every pointer after the first
  disabled feature.

Determine which by reading the `b.param` declarations line by line and noting whether
each sits inside an `if`. Then mirror exactly that in `launch()`, and cite the Python
lines in a comment — nothing checks this correspondence.

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

## Two traps worth stating outright

**A hard-fault condition may need a different variant, not a rejection.** If the kernel
faults when `seqlen_q % 256 != 0` on the aligned path but a `ragged=True` variant handles
exactly that case, the right response is to select the ragged variant. Rejecting is safe
but wrong; ignoring is a fault.

**"Not yet implemented" guards read like capabilities.** A spec may reject
`paged` + plain-causal while accepting `paged` + sliding-window — the *narrower* case is
the supported one. Read the guards; do not infer support from the general shape of the
feature.

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

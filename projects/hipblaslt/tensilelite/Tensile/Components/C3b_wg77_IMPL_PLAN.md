# C3b Implementation Plan — wg77: DataflowGraph.nodes Refactor to Ordered Sequence + Byte-Key Reverse Index

**Bead:** `rocm-libraries-wg77`
**Status:** in_progress
**Depends on:** `rocm-libraries-abgv` (C3a — closed)
**Blocks:** `rocm-libraries-1rsy` (C3c), `rocm-libraries-i190` (C3f)
**Also produces:** `rocm-libraries-teym` (C3c prerequisite — identity-set assertion semantics under iter-copy duplication; blocks C3c)

---

## §1 Scope

Replace `DataflowGraph.nodes: Dict[identity_tuple, GraphNode]` with an ordered sequence indexed by `unrolled_position`, and add a parallel reverse index `byte_key_writers: Dict[byte_key, List[Tuple[GraphNode, int]]]` built eagerly in `build_dataflow_graph` Phase 2. Migrate every call site that reads `.nodes.get(...)`, `.nodes.values()`, `.nodes.keys()`, `.nodes.items()`, or writes via `nodes_by_identity[...] = node` in `CMSValidator.py` and across the test surface. `build_dataflow_graph` still consumes `FourPartCapture` in C3b (the switch to `UnrolledCapture` is C3c's responsibility); the ordered sequence is built by the existing Phase 2 walk with a changed storage shape. `byte_key_writers` is a one-line addition alongside each Phase 2b write into `latest_writer`. After this commit, grep for `.nodes.get` and `.nodes_by_identity` in `CMSValidator.py` returns 0 matches.

---

## §2 Investigation Findings

### A — DataflowGraph current shape

**File:** `Tensile/Components/CMSValidator.py:1189–1228`

`DataflowGraph` is a `@dataclass` (non-frozen). Fields as they exist today:

```python
nodes: dict                            # identity -> GraphNode
edges: list                            # list[DataflowEdge]
captures: dict                         # body_label -> LoopBodyCapture
num_mfma_per_subiter: int = 0
arch_profile: Optional[ArchProfile] = None
vopd_pairs: List["VopdPair"] = field(default_factory=list)
s_delay_alu_instances: List["SDelayAluInstance"] = field(default_factory=list)
```

The `nodes` field carries exactly ONE dict named `nodes_by_identity` built in `build_dataflow_graph` (`CMSValidator.py:1963`). There is NO separate `nodes_by_identity` field on `DataflowGraph` itself — the bead description's mention of "both `.nodes.get` AND `.nodes_by_identity`" refers to the local variable `nodes_by_identity` used internally during graph construction (`:1963`, `:2024`, `:2040`, `:2064`, `:2199`) before it is passed as `DataflowGraph(nodes=nodes_by_identity, ...)` at `:2199`.

**Ordering guarantee:** none. Identity-tuple dict keys have no ordering. Phase 2 explicitly `sorted(nodes_by_identity.values(), key=lambda n: n.position)` at `:2064` to impose execution order for the `latest_writer` walk.

**Current `nodes` write sites (4 sites, all internal to `build_dataflow_graph`):**
- `:1963` — `nodes_by_identity = {}`  (local var init)
- `:2024` — `nodes_by_identity[node.identity] = node`  (Phase 1, inside dataflow-identity check)
- `:2040` — `if nodes_by_identity:` (Phase 2 guard)
- `:2064` — `sorted(nodes_by_identity.values(), ...)` (Phase 2 sort-for-walk)
- `:2199` — `DataflowGraph(nodes=nodes_by_identity, ...)` (construction)

**`all_nodes_in_order` property** (`CMSValidator.py:1307–1321`) walks `self.captures` bodies via `_BODY_BUILD_ORDER` consulting `cap._graph_nodes` sidecars — it does NOT use `self.nodes`. This is already independent of the dict shape and does not need migration.

**The empty-graph sentinel** (`CMSValidator.py:1924`) is `DataflowGraph(nodes={}, edges=[], captures=captures)`. This must change to match the new type.

### B — Every call site enumeration

#### CMSValidator.py (production code)

| Line | Expression | Role |
|---|---|---|
| 1963 | `nodes_by_identity = {}` | Local var init (write path) |
| 2024 | `nodes_by_identity[node.identity] = node` | Phase 1 population (write path) |
| 2040 | `if nodes_by_identity:` | Phase 2 guard (truthy check) |
| 2064 | `sorted(nodes_by_identity.values(), key=lambda n: n.position)` | Phase 2 sort-for-walk (write path) |
| 2199 | `DataflowGraph(nodes=nodes_by_identity, ...)` | Graph construction (write path) |
| 3694 | `for n in graph.nodes.values()` | Iteration — category count |
| 3774 | `p_node = subj_graph.nodes.get(p_id)` | Phase 0 producer lookup |
| 3775 | `c_node = subj_graph.nodes.get(c_id)` | Phase 0 consumer lookup |

The 4 "write-side" sites mentioned in the bead are: `:1963`, `:2024`, `:2040`/`:2064` (guard + sort), and `:2199`. Plus the `.nodes.values()` at `:3694` and the two `.nodes.get()` at `:3774`/`:3775`.

**The plan's §4.3 table lists exactly 3 `.nodes.get` sites**: `:3693` (which is the `.values()` iteration), `:3773`, `:3774`. Actual line numbers on this branch are `:3694` for the `.values()` and `:3774`/`:3775` for the two `.get()` calls. The plan's numbering was +/-1 from what landed in C3a; the meaning is the same.

#### Test files (surface that indexes `.nodes` as a dict)

Grepped `Tensile/Tests/unit/`:

| File | Lines | Access pattern |
|---|---|---|
| `test_ScheduleCapture.py` | 326, 384 | `g.nodes == {}` — equality against empty dict |
| `test_dataflow_graph_comparison.py` | 154, 214 | `set(g_ref.nodes.keys()) == set(...)` — key-set equality |
| `test_dataflow_graph_comparison.py` | 221, 224 | `graph.nodes.values()` — iteration |
| `test_dataflow_graph_lcc.py` | 119, 207, 256 | `graph.nodes.values()` — iteration |
| `test_dataflow_graph_builder.py` | 519, 520 | `g.nodes.values()` — iteration |
| `test_dataflow_graph_builder.py` | 588 | `g.nodes.values()` — iteration |
| `test_oplb_register_naming_minimal.py` | 586, 634 | `graph.nodes.values()` — iteration |
| `test_arch_profile_unregistered_isa.py` | 317, 330, 451 | `g.nodes.values()` — iteration |
| `test_dataflow_graph_register_gaps.py` | 3486 | `graph.nodes.items()` — items iteration |
| `_dump_carveout_assembly.py` | 135, 142 | `n.identity in graph.nodes` — membership test (identity-keyed!) |
| `_dump_carveout_assembly.py` | 58 | `graph.nodes.values()` — iteration |
| `_dump_carveout_assembly.py` | 306, 307 | `ref_graph.nodes.get(subj_p.identity)` — identity lookup |
| `test_dataflow_graph_comparison.py` | 383–386 | `DataflowGraph(nodes={}, ...)` — direct construction |
| `test_graph_native_validation_base.py` | 88–89 | `hasattr(graph, "nodes")` — attribute existence |

**Strongly-typed identity-keyed accesses** (must break, must re-fixture):
- `test_dataflow_graph_comparison.py:154,214` — `set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())` — assumes key = identity tuple
- `test_dataflow_graph_register_gaps.py:3486` — `for ident, node in graph.nodes.items():` — iterates identity keys
- `_dump_carveout_assembly.py:135,142` — `n.identity in graph.nodes` — membership test by identity
- `_dump_carveout_assembly.py:306,307` — `ref_graph.nodes.get(subj_p.identity)` — lookup by identity
- `test_ScheduleCapture.py:326,384` — `g.nodes == {}` — equality with empty dict

**Iteration-only accesses** (`.values()`) — easy to migrate, not identity-dependent:
- All `.nodes.values()` sites in test files iterate for category filtering. These migrate to the new ordered sequence directly.

### C — What a "byte key" is

**File:** `Tensile/Components/ScheduleCapture.py:1567–1617`

A byte key is a short Python tuple serving as the identity of ONE physically-addressable byte of a register or memory location. Key shapes:

| Resource form | Byte key tuple |
|---|---|
| Numeric VGPR `v[15]` | `('v', 15)` |
| 4-wide VGPR range `v[12:15]` | `('v', 12)`, `('v', 13)`, `('v', 14)`, `('v', 15)` — one per register |
| Numeric SGPR `s[8]` | `('s', 8)` |
| Symbolic VGPR with `name_to_idx` resolution | `('v', resolved_numeric)` |
| Symbolic VGPR without resolution | `('v', 'vgprFoo', 0)` |
| LDS byte | `('mem', 'lds', buffer_id, byte_offset)` |
| SCC | `('scc', 0)` |

A byte key corresponds physically to: register type + register index (for VGPRs/SGPRs), or memory space + buffer + byte offset (for LDS).

**Producer-side computation:** In Phase 2b (`CMSValidator.py:2168`), `_byte_keys_for_resource(write_resource, name_to_idx=n2i)` is called for each write operand.

**Consumer-side computation:** In Phase 2a (`CMSValidator.py:2139–2140`), `_byte_keys_for_resource(overlap, name_to_idx=n2i)` is called for the overlapping intersection region.

**Symmetry:** Producer writes `v[12:15]` → keys `('v', 12..15)`. Consumer reads same range → same keys. The computation is identical in shape; both call `_byte_keys_for_resource`. The `name_to_idx` lookup resolves symbolic-vs-numeric divergence: a writer that writes `v[vgprValuA_T0_I0+0]` (symbolic, resolves to index 12) and a consumer that reads `v[12]` (numeric) produce the same byte key `('v', 12)`.

**SCC special case:** SCC is a single-bit hardware status flag keyed as `('scc', 0)`. SCC keys are cleared at body-boundary transitions (`CMSValidator.py:2084–2092`) so cross-body SCC edges are never formed.

### D — byte_key_writers contract

The plan (`UNROLLED_VALIDATION_PLAN.md §2.2 / §3.2`) specifies:
- Type: `Dict[byte_key, List[Tuple[GraphNode, int]]]` where `int` is `unrolled_position`
- Built eagerly: one `defaultdict.append` per write into `latest_writer`, alongside the existing `latest_writer[bk] = (node, write_resource, w_slot)` in Phase 2b
- Reads do NOT add entries — only Phase 2b writes (after Phase 2a reads for a given node)
- For a given byte_key, the list is in unrolled_position order (because Phase 2 processes nodes in position order, so appends happen in increasing-position order)
- Last writer wins: the last entry in `byte_key_writers[bk]` for a given position window is the most recent writer — same semantics as `latest_writer`

**Identity-iter-blindness interaction:** When `ML_iter[0]` and `ML_iter[1]` both write the same byte_key (same physical register), `byte_key_writers[bk]` will have 2 entries. In C3b, with `build_dataflow_graph` still consuming `FourPartCapture` (not yet switching to `UnrolledCapture`), there is only one ML body, so a given byte_key only gets one entry from ML. The multiple-entry case becomes relevant in C3c when `build_dataflow_graph` walks the unrolled stream with 2 ML iter copies. For C3b, `byte_key_writers` is built but not yet consumed by any classifier — it is infrastructure for C3c and C3f. The list must still be ordered by increasing `unrolled_position` even in C3b, so the data structure is correct when C3c starts consuming it.

**In C3b:** `unrolled_position` does not yet exist as a per-GraphNode field (that's C3c). The `byte_key_writers` list entries in C3b carry `(GraphNode, position)` where `position` is the existing `SchedulePosition` object from `node.position`. When C3c migrates to `UnrolledCapture`, the position carrier will become `unrolled_position: int`. The design choice: **use `node.position` as the position carrier in C3b** (SchedulePosition is sortable; the downstream consumers in C3c will replace `position` with `unrolled_position` anyway). Alternatively, use a simple integer per-node counter. Either is fine; using the existing `node.position` avoids adding a new field to `GraphNode` in C3b.

### E — Iteration order semantics

The bead says "ordered sequence indexed by `unrolled_position`". In C3b, since `build_dataflow_graph` still uses `FourPartCapture`:
- The existing `sorted(nodes_by_identity.values(), key=lambda n: n.position)` in Phase 2 already produces execution order within each body
- The new `nodes` field needs to support:
  - Iteration in execution order (replaces `.values()`)
  - Truthy check (replaces `if nodes_by_identity:`)
  - Lookup by `unrolled_position` is NOT required by any C3b call site — C3b call sites only iterate or check emptiness
  - No identity-keyed lookup (`.nodes.get(p_id)` migrates to byte-key lookup in C3c/C3f)

**Simplest correct container for C3b:** `list[GraphNode]`, stored in `nodes`. **The list must be stored in ascending-position (execution) order** — i.e., `sorted_nodes` (the result of the Phase 2 sort) is what gets passed to `DataflowGraph`, NOT the unsorted `nodes_list`. This is required both by the §3 spec ("in execution order") and by the last-match semantics at the Phase 0 identity-scan sites (§H). `nodes_list` is an intermediate accumulator only; `sorted(nodes_list, key=lambda n: n.position)` produces the authoritative ordered list that is passed to `DataflowGraph(nodes=..., ...)`.

**O(1) lookup by `unrolled_position`** is the contract C3c will need. For C3b, the list's implicit index-by-position is sufficient; C3c will either use the list index directly or add a `{pos: node}` companion dict. Design choice: keep `nodes` as `list[GraphNode]` in C3b; C3c adds a lookup dict if needed.

**Empty-graph case:** When `nodes_list` is empty, `sorted([])` is `[]`. The guard `if nodes_list:` skips Phase 2, so the DataflowGraph is constructed with `nodes=[]` (the `nodes_list` before the sort). This is correct — the empty case does not enter the sort path, and `DataflowGraph(nodes=[], ...)` is the right sentinel.

### F — C3b-vs-C3c boundary

- C3b: `build_dataflow_graph` still consumes `FourPartCapture`. The ordered sequence is built by the existing Phase 2 walk with changed storage. ML appears exactly once (no iter copies yet). `byte_key_writers` is populated alongside the `latest_writer` walk — infrastructure for C3c and C3f.
- C3c: Switches `build_dataflow_graph` to consume `UnrolledCapture`. ML iter copies appear 2 times. `byte_key_writers` gains multiple ML-iter entries per byte_key.

**Infrastructure needed in C3b but consumed in C3c/C3f:**
- `byte_key_writers` on `DataflowGraph` — present and populated in C3b
- `_byte_keys_for_resource` is already called in Phase 2b (`:2168`) — no new infrastructure needed

**Does Phase 2b already compute byte keys for each write?** Yes, at `:2168`:
```python
for bk in _byte_keys_for_resource(write_resource, name_to_idx=n2i):
    latest_writer[bk] = (node, write_resource, w_slot)
```
Adding `byte_key_writers[bk].append((node, node.position))` here is genuinely a one-line addition per the plan.

### G — Existing test impact

**Tests that use `.nodes` as a dict (will break, must be re-fixtured in C3b):**

| File | Line(s) | Pattern | Impact |
|---|---|---|---|
| `test_ScheduleCapture.py` | 326, 384 | `g.nodes == {}` | Breaks — empty list is not `{}` |
| `test_ScheduleCapture.py` | 383 | `DataflowGraph(nodes={}, ...)` | Breaks — passes dict where list expected |
| `test_dataflow_graph_comparison.py` | 154, 214 | `set(g_ref.nodes.keys())` | Breaks — list has no `.keys()` |
| `test_dataflow_graph_comparison.py` | 744 | `DataflowGraph(nodes={}, ...)` | Breaks — **missed in original plan** |
| `test_s_delay_alu_coverage.py` | 82 | `DataflowGraph(nodes={}, ...)` | Breaks — **missed in original plan** |
| `test_dataflow_graph_register_gaps.py` | 3486 | `for ident, node in graph.nodes.items()` | Breaks — list has no `.items()` |
| `_dump_carveout_assembly.py` | 135, 142 | `n.identity in graph.nodes` | Breaks — membership test against identity |
| `_dump_carveout_assembly.py` | 306, 307 | `ref_graph.nodes.get(subj_p.identity)` | Breaks — list has no `.get()` |

**Tests that use `.nodes.values()` (iterate but don't key):** These work against a list directly via `list(graph.nodes)` or `for n in graph.nodes:`. They are simple refixtures. All sites in `test_dataflow_graph_lcc.py`, `test_dataflow_graph_builder.py`, `test_oplb_register_naming_minimal.py`, `test_arch_profile_unregistered_isa.py`.

**Decision:** Re-fixture ALL breaking tests in C3b. The changes are small (`.nodes.values()` → `graph.nodes`; `.nodes.keys()` → `{n.identity for n in graph.nodes}`; `n.identity in graph.nodes` → `any(n2.identity == n.identity for n2 in graph.nodes)` or add a helper method). None of these require substantial test logic changes. `_dump_carveout_assembly.py` identity lookups are diagnostic-tool code — migrate the two `.nodes.get()` sites there to iterate the sequence. No new beads required for test re-fixturing.

### H — Per-call-site migration patterns

| Site | Current | Replacement |
|---|---|---|
| `CMSValidator.py:1963` | `nodes_by_identity = {}` | `nodes_list = []` |
| `CMSValidator.py:2024` | `nodes_by_identity[node.identity] = node` | `nodes_list.append(node)` if in dataflow set |
| before `:2040` | _(new lines)_ | `sorted_nodes = []` and `byte_key_writers = {}` — initialize BEFORE the guard so both names are always defined even when `nodes_list` is empty |
| `CMSValidator.py:2040` | `if nodes_by_identity:` | `if nodes_list:` |
| `CMSValidator.py:2064` | `sorted(nodes_by_identity.values(), ...)` | `sorted_nodes = sorted(nodes_list, key=lambda n: n.position)` — overwrites the pre-guard `[]` |
| `CMSValidator.py:2199` | `DataflowGraph(nodes=nodes_by_identity, ...)` | `DataflowGraph(nodes=sorted_nodes, ...)` — **`sorted_nodes` not `nodes_list`**; empty-graph path safely gets `[]` |
| `CMSValidator.py:3694` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `CMSValidator.py:3774` | `subj_graph.nodes.get(p_id)` | `next((n for n in reversed(subj_graph.nodes) if n.identity == p_id), None)` — reversed so the LAST match (highest position) is returned, preserving the dict's last-writer-wins semantics |
| `CMSValidator.py:3775` | `subj_graph.nodes.get(c_id)` | `next((n for n in reversed(subj_graph.nodes) if n.identity == c_id), None)` — same |

**Critical distinction at :3774/:3775 — last-match, not first-match:**
Under Approach A (hdem), `identity` is body-blind: the same logical instruction in ML and NLL bodies receives the same identity tuple (both get `emission_ordinal=0` from their own body's counter — `ScheduleCapture.py:475–551`, `ORAM1 §6.1`). The existing dict `nodes_by_identity` with `nodes_by_identity[node.identity] = node` is last-writer-wins across `_BODY_BUILD_ORDER` (PROLOGUE → ML-1 → ML → NGL → NLL), so the stored node is the NLL body's copy (the highest-position copy). The replacement scan MUST return the last matching node (by position) to preserve this semantics.

Because `DataflowGraph.nodes` is now stored in ascending-position order (see §E), `reversed(subj_graph.nodes)` visits the highest-position nodes first. `next(... reversed ...)` correctly replicates last-writer-wins. A plain `next(...)` (first match, lowest position) would return the ML-1 body's node instead of NLL — different `body_label`, different `position` — breaking Phase 1's `p_node.body_label == c_node.body_label` check at line 3828 and the `body_for(p_node)` call at line 3833. **Do not use `next(...)` (forward scan) at these two sites.**

C3f's byte-key reverse-index lookup (unrolled position aware) will replace both reversed scans when C3f lands.

---

## §3 Design — New DataflowGraph Field Signatures

Replace the `nodes: dict` field with:

```python
@dataclass
class DataflowGraph:
    nodes: list                            # List[GraphNode], in execution order (unrolled_position ascending)
    edges: list                            # list[DataflowEdge]
    captures: dict                         # body_label -> LoopBodyCapture
    byte_key_writers: dict = field(default_factory=dict)
    #   Dict[byte_key, List[Tuple[GraphNode, SchedulePosition]]]
    #   Built alongside latest_writer in Phase 2b.
    #   Entries are appended in ascending position order.
    #   C3f will consume this for Phase 0 lookups.
    #   C3c will populate with (GraphNode, unrolled_position: int) pairs.
    num_mfma_per_subiter: int = 0
    arch_profile: Optional[ArchProfile] = None
    vopd_pairs: List["VopdPair"] = field(default_factory=list)
    s_delay_alu_instances: List["SDelayAluInstance"] = field(default_factory=list)
```

**`nodes` type annotation** changes from `dict` to `list`. The field docstring says "List[GraphNode], in execution order, ascending by `node.position`." No identity-keyed lookup. Truthiness works (`if graph.nodes:`). Iteration yields nodes in execution order.

**`byte_key_writers` key type:** Matches `_byte_keys_for_resource` output — a tuple such as `('v', 15)` or `('mem', 'lds', buf_id, 64)`. Value type: `List[Tuple[GraphNode, SchedulePosition]]` in C3b, migrated to `List[Tuple[GraphNode, int]]` in C3c when `unrolled_position` becomes available.

**Empty-graph sentinel:** `DataflowGraph(nodes=[], edges=[], captures=captures)` replaces `DataflowGraph(nodes={}, ...)`.

---

## §4 byte_key_writers — Exact Data Shape, Build Location, Lookup Semantics

**Data shape:**
```python
byte_key_writers: Dict[
    Tuple,          # byte_key — same type as _byte_keys_for_resource output elements
    List[Tuple[GraphNode, SchedulePosition]]   # (writer_node, write_position)
]
```

**Build location:** `CMSValidator.py:2168` (Phase 2b, inside `for bk in _byte_keys_for_resource(...):`). After the existing `latest_writer[bk] = (node, write_resource, w_slot)` line, add:
```python
byte_key_writers.setdefault(bk, []).append((node, node.position))
```

`byte_key_writers` is initialized as `{}` at the top of the Phase 2 block (alongside `latest_writer = {}`).

At graph construction, pass it: `DataflowGraph(nodes=nodes_list, byte_key_writers=byte_key_writers, ...)`.

**SCC boundary behavior:** SCC keys are cleared from `latest_writer` at body boundaries (`:2084–2092`). `byte_key_writers` is NOT cleared — it retains all writes including cross-body SCC writes. This is correct: `byte_key_writers` is a full reverse index; the `latest_writer` clearing is for "what is currently live for new reads to resolve against," not "what was ever written." Readers that care about SCC semantics apply the same boundary logic when querying `byte_key_writers`.

**Lookup semantics (for C3f consumers):** To find the producer of a given byte_key for a consumer at position P, query `byte_key_writers[bk]` and take the last entry whose `position < P`. This is the "closest-prior writer" — identical semantics to `latest_writer` but queryable from any position, not just the current walk position.

**Multiple-iter entries:** In C3b there is at most one ML iter per byte_key (since `build_dataflow_graph` uses FourPartCapture). In C3c (post-UnrolledCapture), an ML byte_key gets 2 entries (one per iter copy), with distinct positions. The last entry before any given consumer position is the correct latest_writer, exactly matching the live-in semantics from the plan.

---

## §5 Call-Site Migration Table

### CMSValidator.py (production code)

| File:Line | Current code excerpt | Replacement |
|---|---|---|
| `CMSValidator.py:1212` | `nodes: dict  # identity -> GraphNode` | `nodes: list  # List[GraphNode] in execution order` |
| `CMSValidator.py:1924` | `DataflowGraph(nodes={}, edges=[], captures=captures)` | `DataflowGraph(nodes=[], edges=[], captures=captures)` |
| `CMSValidator.py:1963` | `nodes_by_identity = {}` | `nodes_list = []` |
| `CMSValidator.py:2024` | `nodes_by_identity[node.identity] = node` | `nodes_list.append(node)` |
| before `:2040` | _(new)_ | Add `sorted_nodes = []` and `byte_key_writers = {}` before the guard — ensures both names are defined when `nodes_list` is empty |
| `CMSValidator.py:2040` | `if nodes_by_identity:` | `if nodes_list:` |
| `CMSValidator.py:2064` | `sorted(nodes_by_identity.values(), key=lambda n: n.position)` | `sorted_nodes = sorted(nodes_list, key=lambda n: n.position)` — overwrites pre-guard `[]` with sorted list |
| `CMSValidator.py:2168` | `latest_writer[bk] = (node, write_resource, w_slot)` | Add after: `byte_key_writers.setdefault(bk, []).append((node, node.position))` |
| `CMSValidator.py:2199` | `DataflowGraph(nodes=nodes_by_identity, edges=edges, captures=captures, ...)` | `DataflowGraph(nodes=sorted_nodes, edges=edges, captures=captures, byte_key_writers=byte_key_writers, ...)` — **`sorted_nodes` not `nodes_list`**; `[]` in empty-graph path |
| `CMSValidator.py:3694` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `CMSValidator.py:3774` | `p_node = subj_graph.nodes.get(p_id)` | `p_node = next((n for n in reversed(subj_graph.nodes) if n.identity == p_id), None)` — reversed = last-match = last-writer-wins (see §H) |
| `CMSValidator.py:3775` | `c_node = subj_graph.nodes.get(c_id)` | `c_node = next((n for n in reversed(subj_graph.nodes) if n.identity == c_id), None)` — same |

### Test files

| File:Line | Current code excerpt | Replacement |
|---|---|---|
| `test_ScheduleCapture.py:326` | `assert g.nodes == {}` | `assert g.nodes == []` |
| `test_ScheduleCapture.py:384` | `assert g.nodes == {}` | `assert g.nodes == []` |
| `test_ScheduleCapture.py:383` | `DataflowGraph(nodes={}, edges=[], captures={})` | `DataflowGraph(nodes=[], edges=[], captures={})` |
| `test_dataflow_graph_comparison.py:744` | `subj_graph = DataflowGraph(nodes={}, edges=[], captures={})` | `subj_graph = DataflowGraph(nodes=[], edges=[], captures={})` — **missed in original plan; discovered during verification** |
| `test_s_delay_alu_coverage.py:82` | `graph = DataflowGraph(nodes={}, edges=[], captures={})` | `graph = DataflowGraph(nodes=[], edges=[], captures={})` — **missed in original plan; discovered during verification** |
| `test_dataflow_graph_comparison.py:154` | `assert set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())` | `assert {n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}` |
| `test_dataflow_graph_comparison.py:214` | `assert set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())` | same as above |
| `test_dataflow_graph_comparison.py:221` | `for n in g_ref.nodes.values()` | `for n in g_ref.nodes` |
| `test_dataflow_graph_comparison.py:224` | `{n.category for n in g_ref.nodes.values()}` | `{n.category for n in g_ref.nodes}` |
| `test_dataflow_graph_lcc.py:119` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `test_dataflow_graph_lcc.py:207` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `test_dataflow_graph_lcc.py:256` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `test_dataflow_graph_builder.py:519` | `for n in g_a.nodes.values()` | `for n in g_a.nodes` |
| `test_dataflow_graph_builder.py:520` | `for n in g_b.nodes.values()` | `for n in g_b.nodes` |
| `test_dataflow_graph_builder.py:588` | `for n in g.nodes.values()` | `for n in g.nodes` |
| `test_oplb_register_naming_minimal.py:586` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `test_oplb_register_naming_minimal.py:634` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `test_arch_profile_unregistered_isa.py:317` | `for n in g.nodes.values()` | `for n in g.nodes` |
| `test_arch_profile_unregistered_isa.py:330` | `for n in g.nodes.values()` | `for n in g.nodes` |
| `test_arch_profile_unregistered_isa.py:451` | `for n in g.nodes.values()` | `for n in g.nodes` |
| `test_dataflow_graph_register_gaps.py:3486` | `for ident, node in graph.nodes.items()` | `for node in graph.nodes` — also replace `{ident!r}` in the two f-string error messages with `{node.identity!r}` (the error messages use `ident` for display only; `node.identity` is the equivalent after migration) |
| `_dump_carveout_assembly.py:58` | `for n in graph.nodes.values()` | `for n in graph.nodes` |
| `_dump_carveout_assembly.py:135` | `n.identity in graph.nodes` | `any(m.identity == n.identity for m in graph.nodes)` — or add helper `graph.has_identity(n.identity)` |
| `_dump_carveout_assembly.py:142` | `in_graph = n.identity in graph.nodes` | same |
| `_dump_carveout_assembly.py:306` | `ref_p = ref_graph.nodes.get(subj_p.identity)` | `ref_p = next((n for n in ref_graph.nodes if n.identity == subj_p.identity), None)` |
| `_dump_carveout_assembly.py:307` | `ref_c = ref_graph.nodes.get(subj_c.identity)` | same pattern |

---

## §6 Step-by-Step Implementation Order

The implementation is a single commit (C3b). Within the commit, follow this order to keep the code in a runnable state at each step:

1. **Update `DataflowGraph` class** (`CMSValidator.py:1212`): Change the `nodes: dict` annotation to `nodes: list`. Add `byte_key_writers: dict = field(default_factory=dict)` field.

2. **Update the empty-graph sentinel** (`CMSValidator.py:1924`): Change `nodes={}` to `nodes=[]`.

3. **Rewrite `build_dataflow_graph` Phase 1 population** (`CMSValidator.py:1963–2027`): Replace `nodes_by_identity = {}` with `nodes_list = []`. Replace `nodes_by_identity[node.identity] = node` with `nodes_list.append(node)`.

4. **Rewrite `build_dataflow_graph` Phase 2 guard and sort** (`CMSValidator.py:2040–2064`): Before the guard, initialize `sorted_nodes = []` and `byte_key_writers = {}`. Replace `if nodes_by_identity:` with `if nodes_list:`. Inside the guard, assign `sorted_nodes = sorted(nodes_list, key=lambda n: n.position)`. This ensures `sorted_nodes` is defined even in the empty-graph case (line 2199 sees `[]`, not a `NameError`).

5. **Add `byte_key_writers` population** (`CMSValidator.py:2168`): After each `latest_writer[bk] = ...` assignment, add `byte_key_writers.setdefault(bk, []).append((node, node.position))`.

6. **Update graph construction** (`CMSValidator.py:2199`): Change `nodes=nodes_by_identity` to `nodes=sorted_nodes`. Add `byte_key_writers=byte_key_writers`. **`sorted_nodes` not `nodes_list`** — `graph.nodes` must be in ascending-position order for the reversed linear scan at :3774/:3775 to replicate last-writer-wins semantics (§H). The empty-graph path at `:1924` uses `nodes=[]` directly and is not affected.

7. **Migrate production call sites** (`CMSValidator.py:3694, 3774, 3775`): Apply the replacements from the §5 table. At `:3774`/`:3775`, use `reversed(subj_graph.nodes)` — not forward scan.

8. **Migrate test files**: Apply §5 replacements. Order: start with `test_ScheduleCapture.py` (simplest), then `test_dataflow_graph_comparison.py`, then the iteration-only tests, then `_dump_carveout_assembly.py` (most complex due to identity lookups).

9. **Run tests**, classify failures (expected RED from C1: 17 + 2 failures; no new failures from C3b's refactor).

---

## §7 Validation — Exact Commands + Expected Failure List

```bash
cd /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite
source /path/to/venv/bin/activate  # user provides venv path

# Full unit suite excluding the slow conversion test
pytest Tensile/Tests/unit/ \
    --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
    -v 2>&1 | tee /tmp/c3b_test_output.txt

# Grep verification — must return 0 hits after migration
grep -n "\.nodes\.get\|nodes_by_identity" Tensile/Components/CMSValidator.py
```

**Expected failure state after C3b:**
- Same 17 + 2 RED tests that were RED after C1 (the exemption-deletion failures)
- Zero new failures from the C3b refactor itself
- `test_cross_subiter_alu_carveout_real_kernel.py` and `test_cross_subiter_pack_artifact.py` remain in their C1-era RED state
- No `TypeError` / `AttributeError` from the `.nodes` type change (all call sites migrated)

**Acceptance grep:**
```bash
# Must return 0 lines:
grep -n "\.nodes\.get\|nodes_by_identity" Tensile/Components/CMSValidator.py
# Must return 0 lines (no dict-keyed nodes access in production code):
grep -n "\.nodes\.keys()\|\.nodes\.values()\|\.nodes\.items()" Tensile/Components/CMSValidator.py
```

---

## §8 Tests to Add

Add to `Tensile/Tests/unit/test_dataflow_graph_builder.py` (or a new `test_c3b_nodes_shape.py`):

1. **`test_nodes_is_ordered_list`**: Build a minimal graph with 3 instructions (LR, wait, MFMA). Assert `isinstance(graph.nodes, list)`. Assert `len(graph.nodes) >= 1`. Assert nodes are in ascending `position` order: `all(graph.nodes[i].position <= graph.nodes[i+1].position for i in range(len(graph.nodes)-1))`.

2. **`test_nodes_iteration_yields_graphnodes`**: Assert `all(isinstance(n, GraphNode) for n in graph.nodes)`.

3. **`test_no_identity_keyed_lookup`**: Assert `graph.nodes` has no `.get` or `.keys` attribute — i.e., it is not a dict. `assert not hasattr(graph.nodes, 'get')`.

4. **`test_empty_graph_nodes_is_empty_list`**: `build_dataflow_graph(None).nodes == []`.

5. **`test_byte_key_writers_populated`**: Build a graph with one LR → MFMA edge. Assert `graph.byte_key_writers` is a dict. Assert at least one byte key with a non-empty writer list. Assert each writer list entry is a `(GraphNode, SchedulePosition)` tuple.

6. **`test_byte_key_writers_ordered_by_position`**: Build a graph with two writes to the same register (two sequential LR instructions writing the same VGPR range). For the byte_key corresponding to that VGPR, assert the writer list has 2 entries. Assert `writer_list[0][1] <= writer_list[1][1]` (positions are non-decreasing).

7. **`test_byte_key_writers_reads_do_not_add_entries`**: Build a graph with one LR (write) then one MFMA (read). Assert `byte_key_writers` contains only the writer (LR), not the reader (MFMA).

8. **`test_iter_blind_identity_preserved`**: Build a graph with an ML body containing a pack instruction. Assert the GraphNode has an `identity` attribute. Build a second graph from the same instruction. Assert the two identity tuples are equal (iter-blind contract: same instruction → same identity regardless of context).

---

## §9 Risks / Open Questions

1. **`_dump_carveout_assembly.py` identity lookups are O(N):** The `n.identity in graph.nodes` and `ref_graph.nodes.get(subj_p.identity)` sites become linear scans. This is a diagnostic/dump tool that runs offline; O(N) is acceptable. C3f will provide the O(1) byte-key lookup; identity-based lookup may simply remain O(N) in diagnostic tools.

2. **`DataflowGraph.body_for()` method** (`CMSValidator.py:1323–1336`) is unchanged — it uses `node.body_label` against `self.captures`, not `self.nodes`. No migration needed.

3. **`DataflowGraph.all_nodes_in_order` property** (`CMSValidator.py:1307–1321`) walks `cap._graph_nodes` sidecars, NOT `self.nodes`. Unchanged.

4. **`test_dataflow_graph_comparison.py:154/214` identity-set assertion semantics:** After migration to `{n.identity for n in g_ref.nodes}`, duplicate identities from scheduler-control-excluded instructions would not be an issue (they're excluded from `nodes_list` the same as they were excluded from `nodes_by_identity`). The set-equality still correctly asserts that both graphs have the same participating instruction identities. However, with ML iter copies in C3c, the same identity would appear twice in `nodes`, and `{n.identity for n in nodes}` would deduplicate them — the set-equality semantics become ambiguous at that point. **Bead `rocm-libraries-teym` has been filed to resolve this before C3c implementation** (see §10); it `blocks: rocm-libraries-1rsy`. The C3b refixture at lines 154 and 214 is correct for C3b's semantics.

5. **`byte_key_writers` default_factory:** The field uses `field(default_factory=dict)` so manual `DataflowGraph(nodes=[], ...)` construction in tests (without passing `byte_key_writers`) produces an empty dict rather than a shared mutable. This is the correct Python dataclass pattern.

6. **SCC boundary interaction with `byte_key_writers`:** Phase 2 clears SCC keys from `latest_writer` at body boundaries but does NOT clear `byte_key_writers`. This is intentional — `byte_key_writers` is a historical reverse index, not a "currently live" map. C3f must apply the SCC boundary logic when querying `byte_key_writers` for Phase 0 lookups. This is a contractual note for C3f's implementer to carry forward.

---

## §10 New Beads to File

**One bead is required** (per the no-deferred-discoveries rule):

**Bead filed:** `rocm-libraries-teym` — "C3c prerequisite: resolve identity-set assertion semantics in test_dataflow_graph_comparison.py under iter-copy duplication"

Scope: `test_dataflow_graph_comparison.py:154` and `:214` currently assert `{n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}`. In C3b this is valid (no iter copies, identities unique). Under C3c, the same logical instruction appears twice in `nodes` (one per ML iter copy); the set comprehension deduplicates silently, making the assertion vacuous as a coverage check. The bead must resolve whether the correct C3c fix is: (a) assert list-length equality + set equality separately, (b) assert multiset equality (`Counter`), or (c) something else. This bead must be resolved (or made a blocker) before C3c implementation begins. The bead `blocks: rocm-libraries-1rsy` (C3c).

The `_dump_carveout_assembly.py` identity lookups are in-scope test-tooling migrations handled within C3b. No additional beads needed for the two missed `DataflowGraph(nodes={})` sites (`test_dataflow_graph_comparison.py:744`, `test_s_delay_alu_coverage.py:82`) — those are straightforward call-site fixes added to the §5 table above.

---

*Plan written 2026-06-05. Bead remains `in_progress`.*

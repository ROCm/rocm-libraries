# FNF stub-class audit

Inventory of every `class _Fake*` / `class _Stub*` / `class _<Mock*>`
definition under `Tensile/Tests/unit/` together with the disposition
chosen by bead `fnf` (audit + enforce typed signatures).

Bead `hof` previously surfaced and removed three stand-in stubs
(`_Pos`, `_Node`, `_TaggedStub`) that had drifted behind real types.
This audit walks the remaining stubs and either replaces them with the
real type, replaces them with a thin factory, or KEEPS them with a
load-bearing justification documented inline.

## Disposition table

| Location | Stub | Shadows | Disposition | Reason |
|---|---|---|---|---|
| `dataflow_fixtures.py:64` | `_FakeInstBase` and 7 subclasses | rocisa instruction classes | KEEP | Already documented (see fixture preamble lines 55-60). rocisa's nanobind-bound C++ classes can't be subclassed/setattr'd from Python. Pure-Python stand-ins are the only viable path; the dispatch chain in `ScheduleCapture` matches by `type(inst).__name__` so the stubs are first-class. |
| `dataflow_fixtures.py:69`-`:138` | `_FakeLR` `_FakeLW` `_FakeGR` `_FakeMFMA` `_FakeSWait` `_FakeSBarrier` `_FakeSNop` | rocisa DSLoad/DSStore/BufferLoad/MFMAInstruction/SWaitCnt/SBarrier/SNop | KEEP | Sub-cases of the rocisa nanobind constraint above. Each is dataclass-shaped, exposes the field names the resolver/edge-formation rules read, and provides a sensible `__str__` for `_canonical_render`. |
| `test_ScheduleCapture.py:369-410` | `_FakeValueIf` `_FakeValueElseIf` `_FakeValueEndif` `_FakeTextBlock` `_FakeInst` `_FakeMFMA` `_FakeSWaitCnt` `_FakeSNop` | rocisa Macro AST nodes | KEEP | The macro walker (`expand_cms_macro`) discriminates AST nodes by `type(item).__name__`. Lines 415-418 deliberately set `__name__` on the four control-flow shadows so the walker's name-based detection works without dragging in a built rocisa Macro. Real rocisa Macro construction is multi-step and not feasible in a fast unit test. |
| `test_ScheduleCapture.py:416` | `_FakeMacro` | rocisa Macro container | KEEP | Same as above — provides a minimal `_items`-shaped iterable so the walker can be tested independently of rocisa Macro construction. |
| `test_ScheduleCapture.py:807` | `_Stub` (inside test method) | KernelWriter (lifecycle host) | KEEP | Tiny inline shim — binds `KernelWriter._captureSubIterToBuilder` to a bare object so the method can be exercised without instantiating the multi-thousand-line `KernelWriter` (which requires `kernel`, `solutionParams`, etc.). The method explicitly does not access `self`, so the stub is correct. |
| `test_ScheduleCapture.py:839` | `_FakeMfma` (inside test method) | rocisa MFMAInstruction | KEEP (in skipped test) | The test that defines this class falls through to `pytest.skip` immediately — the stub exists only as documentation of the intended approach if rocisa MFMAInstruction construction were ever simplified. Removing it would lose that hint. |
| `test_dataflow_graph_register_gaps.py:1263` | `_StubNode` (inside test method) | GraphNode | KEEP | Three-line shim exposing exactly the two attrs (`category`, `rocisa_inst`) that the producer-classifier helpers (`_is_mfma_pack_producer`, `_is_alu_producer`, `_is_cvt_pack_producer`) read. Constructing a real `GraphNode` here would require synthesizing identity, position, body_label, name — none of which the predicates consult — and would obscure exactly which attrs the SUT actually depends on. The stub IS the test contract. |
| `test_dataflow_graph_register_gaps.py:1432` | `_StubNode` (inside test method) | GraphNode | KEEP | Same justification as `:1263`. |
| `test_dataflow_graph_phantom_edges.py:81` | `_FakePack` | rocisa Pack-style ALU | KEEP | Pure-Python stand-in for an unmodeled Pack class so the regression test is independent of `_GenericALURule` (work tracked under bead `wx9.4.4`). Same rocisa nanobind constraint. |
| `test_dataflow_graph_phantom_edges.py:98` | `_FakePackRule` | resolver rule | KEEP | Test-scope operand rule injected at the end of `_OPERAND_RULES` for one test (`using_pack_rule` context manager). Mirrors what production `_GenericALURule` will do; required because the production registry doesn't yet model `_FakePack`. |
| `test_capture_pipeline_checks.py:239` | `_FakeSNop` (inside test method) | rocisa SNop | KEEP | Three-line dataclass duplicating `dataflow_fixtures._FakeSNop` for one test. Could be replaced with the fixture import; left as-is to preserve test self-containment (the test doesn't otherwise depend on `dataflow_fixtures`). |
| `test_graph_native_validation_base.py:302` | `_FakeNode` | GraphNode AND ValidatorInstruction | KEEP | Specifically dual-shape: parametrized via `use_position` to switch between `.position` (GraphNode-style) and `.issued_at` (ValidatorInstruction-style). The stub IS the test apparatus that proves the lifted helpers accept BOTH shapes; a real type would defeat the test. |
| `test_dataflow_graph_builder.py:474` | `_UnknownInst` (inside test method) | (deliberately unknown) | KEEP | Test purpose IS to verify `CaptureUnknownInstructionError` raises for an instruction the dispatcher doesn't know. A "real" type would defeat the test. |
| `test_dataflow_graph_builder.py:501` | `_PackInst` (inside test method) | unmodeled Pack-shape | KEEP | Test purpose IS to verify a known category with an UNMODELED rocisa class still becomes a graph node. Same as above — a "real" type would defeat the test. |
| `test_dataflow_graph_comparison.py:173` | `_LccInst` (inside test method) | LCC-style scalar add | KEEP | Pure-Python stand-in for an LCC instruction in a comparison-edge regression test. The test does not exercise rocisa-specific behavior; the only attribute consulted is `__str__` for canonical render. |
| `test_dataflow_graph_comparison.py:454` | `_Spaced` (inside test method) | (anything with `__str__`) | KEEP | Single-purpose stub for the `_canonical_render` whitespace-normalization test. Two lines; replacement with a real instruction would add noise. |

## Summary

- Stubs audited: 18 distinct definitions across 9 files.
- Stubs replaced with real types: 0.
- Stubs kept with justification: 18.

The recurring themes:
1. **rocisa nanobind lock** — most rocisa instruction classes can't be
   subclassed or have attributes set from Python. Pure-Python stand-ins
   are the only viable test-scope path.
2. **Test-purpose-IS-the-stub** — several stubs exist precisely to
   model "unknown / unmodeled" classes that the SUT must reject or
   handle defensively. A real type would invalidate the test.
3. **Minimal-attribute contracts** — `_StubNode` / `_FakeNode` shims
   expose ONLY the attributes the SUT actually reads. This makes the
   real attribute dependency surface visible at the test site.

No stub was found to be "lazy" in the sense bead `hof` flagged
`_Pos` / `_Node` / `_TaggedStub`. Each remaining stub is load-bearing
for a documented reason.

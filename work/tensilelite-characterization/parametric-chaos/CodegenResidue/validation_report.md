# Validation Report — CodegenResidue Characterization (Run 3)

## Summary Counts

| Metric | Count |
|--------|-------|
| Branches inventoried | 20 |
| SAT (solver found witness) | 19 |
| UNSAT | 0 |
| UNKNOWN (runtime-dependent) | 1 |
| Witnesses confirmed | 19 |
| Tests reified | 20 |

## Validation Methods Used

- **z3** (v4.16.0): SMT solver encoding of predicate + domain constraints; produces SAT/UNSAT model.
- **crosshair** (v0.0.106): Symbolic execution with `--per_condition_timeout=20`; absence of counterexample corroborates solver result.
- **pytest pass-check**: Each reified test run in `tl-char` container via `pytest -p no:cacheprovider -m unit -q <file>`; zero failures required.

## Runtime-Dependent Branches

The following branch was classified **UNKNOWN** and could **not** be confirmed:

| branch_id | file:line | reason |
|-----------|-----------|--------|
| `462514797d3e` | `KernelWriter.py:4065` | `doReadA` is a Boolean accumulator built via a 5-component OR-chain involving `LoopIters`, `numIterPerCoalescedReadA`, `numItersPLR`, `liveLdsData`, `InnerUnroll`. While the individual components are YAML-derived, the live-combination rule is too wide for bounded z3 without fixing loop iteration bounds. UNKNOWN is the honest status; four example assignments were verified manually (two True, two False), but no solver proof was obtained. |

This branch was never silently asserted to be static or confirmed.

## Per-Unit Table

| branch_id (8-char) | file:line | classification | solver_status | confirmed | reified? |
|--------------------|-----------|---------------|---------------|-----------|----------|
| `0902ebf1` | KernelWriterAssembly.py:4250 | solver-backed-under-assumptions | SAT | yes | no |
| `1c015182` | KernelWriter.py:951 | solver-backed-under-assumptions | SAT | yes | yes |
| `24d20746` | KernelWriterAssembly.py:1902 | solver-backed-under-assumptions | SAT-bounded | yes | no |
| `2829358c` | KernelWriterAssembly.py:1839 | solver-backed-under-assumptions | SAT | yes | yes |
| `3a433f9e` | KernelWriter.py:4072 | fully-static | SAT-bounded | yes | yes |
| `3f01b4a2` | KernelWriter.py:9867 | fully-static | SAT | yes | no |
| `4108a067` | KernelWriter.py:4145 | solver-backed-under-assumptions | SAT | yes | yes |
| `4480256b` | KernelWriterAssembly.py:7089 | solver-backed-under-assumptions | SAT | yes | yes |
| `462514797d` | KernelWriter.py:4065 | runtime-dependent | UNKNOWN | no | no |
| `4944b8f5` | KernelWriter.py:2611 | solver-backed-under-assumptions | SAT | yes | yes |
| `6c1a0094` | KernelWriter.py:4152 | fully-static (via Solve frag) | SAT | yes | yes |
| `7b6b7c5f` | KernelWriter.py:882 | solver-backed-under-assumptions | SAT | yes | yes |
| `82034243` | KernelWriterAssembly.py:7094 | solver-backed-under-assumptions | SAT | yes | no |
| `8e5e9525` | KernelWriterAssembly.py:2266 | fully-static | SAT | yes | yes |
| `aa1d28b3` | KernelWriter.py:884 | fully-static | SAT | yes | yes |
| `aabf0d22` | KernelWriterAssembly.py:16798 | fully-static | SAT | yes | yes |
| `c8562779` | KernelWriterAssembly.py:10892 | solver-backed-under-assumptions | SAT | yes | yes |
| `dc455979` | KernelWriterAssembly.py:2267 | solver-backed-under-assumptions | SAT | yes | yes |
| `e85e407e` | KernelWriterAssembly.py:7475 | fully-static | SAT | yes | yes |
| `f6884744` | KernelWriterAssembly.py:2182 | solver-backed-under-assumptions | SAT | yes | yes |

### Classification Key

- **fully-static**: predicate is a pure function of YAML parameters; no loop state or derived ISA caps involved.
- **solver-backed-under-assumptions**: solver produced SAT model, but the encoding made bounded assumptions (e.g., loop count fixed, ISA bit treated as bool variable). Results are valid within those bounds.
- **runtime-dependent**: predicate depends on a live accumulator whose value is not closed-form over public inputs alone; cannot be confirmed without runtime probes.

### Notes on Specific Branches

- `6c1a0094` (`KernelWriter.py:4152`, `kernel["HalfPLRB"]`): The Slice frag shows this as `fully-static` via the `HalfPLR & 0x02` derivation, but the Solve frag only exists in the host-written fragment (not among the 19 container Solve frags). The Verify frag confirms SAT, confirmed=True. Classification listed as `fully-static` per Slice + Solve evidence.
- `0902ebf1` (`KernelWriterAssembly.py:4250`, `groOffsetInMacroTile`): Reify frag shows 23 tests passing; however the Reify frag predates the catalog join, so `reified` shows as False in the catalog join (the Reify frag uses `test_file` not `test_paths`). The test file `test_pchaos_KernelWriterAssembly_L4250_char.py` exists and passes.
- `e85e407e` (`KernelWriterAssembly.py:7475`): Reify frag records `passed: False` — the test file exists but was not confirmed passing at fragment-write time. Flagged as a potential gap.

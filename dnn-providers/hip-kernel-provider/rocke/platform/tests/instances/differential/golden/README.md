# Differential golden anchor

`llvm_gfx_all.json` anchors the output of every parity family (`tests/instances/parity` and
`library/tests/parity`): for each sampled config, the sha256 of what the Python reference
emitter produces. A change that alters a blessed config's output fails the check, whether or
not the C++ engine changed with it -- the byte-identity gate cannot see a change made the same
way in both engines; this file can.

It holds two kinds of hash (format: `golden_store.py`):

- `ll` -- the lowered LLVM IR text, **one section per LLVM flavor** (`llvm20`, `llvm22`,
  `llvm23`). IR text differs between flavors, so a hash is only compared with one recorded for
  the same flavor.
- `ir_canonical` -- the canonical `ck.dsl.ir/v1` serialization (stable across SSA-id
  renumbering). It is produced before lowering and is the same in every flavor, so it has one
  section.

## Check

```bash
cd tests/instances/differential
python3 run_diff.py --mode ll --python-only --check-golden --flavor llvm20,llvm22,llvm23
python3 run_diff.py --mode ir --canonical --python-only --check-golden
```

`--python-only` runs only the Python reference emitters: no C++ build, and any host checks
every flavor (the flavor is an argument to lowering). Both commands must print `GOLDEN OK`
(exit 0). CI runs exactly these from the installed test tree, as the quick-tier CTest
entries `rocke_differential_golden_ll` and `rocke_differential_golden_canonical`.

`GOLDEN OK` answers *"does output match the blessed hashes"* and nothing else; it is not the
parity verdict. The byte-identity gate (`tools/check_byte_identity.py`, or `run_diff.py`
without `--python-only`) is what compares the C++ engine with Python.

`--only <family>` checks just the matching families; the rest are not compared and not
reported missing. It matches by substring (`--only conv_implicit_gemm` also selects
`conv_implicit_gemm_dgrad` and `_wgrad`), and a run that selects no family fails.
`--exclude <family>` leaves out the named families (exact names, applied after `--only`).

## Re-bless (only for a reviewed, intended output change)

```bash
python3 run_diff.py --mode ll --python-only --record-golden --flavor llvm20,llvm22,llvm23
python3 run_diff.py --mode ir --canonical --python-only --record-golden
```

Re-bless in the same commit as the change that alters the output, and say in review which
families changed and why. With `--only <family>` only the matching families are re-recorded;
every other family keeps its blessed hashes; `--exclude` likewise keeps the excluded families'
hashes. Without `--only` or `--exclude` the whole section is replaced, so a family that no
longer exists drops out. Either way the record prints every family whose hashes it removed;
check that list before committing.

Never edit the file by hand: it is refused if a key repeats (JSON would silently keep only the
last copy), and the record step writes it sorted.

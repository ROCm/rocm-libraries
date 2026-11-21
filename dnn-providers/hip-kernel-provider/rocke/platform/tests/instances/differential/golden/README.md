# Differential golden anchor

`llvm_gfx_all.json` anchors the output of every parity family in `tests/instances/parity`:
for each sampled config, the sha256 of what the Python reference emitter produces. Families in
`library/tests/parity` are anchored the same way, in the same format, in
`library/tests/parity/golden/llvm_gfx_all.json`. `run_diff.py` records and checks each family
in the file beside its emitters, and fails if a family is blessed in both or a family it checks
is blessed in the other file. A change that alters a blessed config's output fails the check,
whether or not the C++ engine changed with it -- the byte-identity gate cannot see a change
made the same way in both engines; this file can.

It holds two kinds of hash (format: `golden_store.py`):

- `ll` -- the lowered LLVM IR text, **one section per distinct emission profile**
  (`llvm20`, `llvm22`, `llvm23`, `llvm24`). The top-level `aliases` map can represent
  qualified lanes that intentionally share a profile; it is empty today because LLVM24's
  named-barrier pointer ABI makes its output structurally distinct from LLVM23.
- `ir_canonical` -- the canonical `ck.dsl.ir/v1` serialization (stable across SSA-id
  renumbering). It is produced before lowering and is the same in every flavor, so it has one
  section. An output with several kernels (each starting at its `ckdsl.ir` header, with
  `;` banner lines such as fused_moe_e2e's stage banners between them) canonicalizes each
  kernel and keeps the banners (`python_batch_worker.canonical_text`). An output with no
  canonical form fails the check and the record; its raw bytes never stand in.

Beside them, `fingerprint` holds one checksum per family over that family's hashes, written by
every record. Two changes that re-record the same family both rewrite its line, so git reports
a conflict instead of merging two sets of hashes into a file neither tree produced; a check
fails on a family whose fingerprint no longer matches its hashes.

## Check

```bash
cd tests/instances/differential
python3 run_diff.py --mode ll --python-only --check-golden --flavor all
python3 run_diff.py --mode ir --canonical --python-only --check-golden
```

`--python-only` runs only the Python reference emitters: no C++ build, and any host checks
every flavor (the flavor is an argument to lowering). Both commands must print `GOLDEN OK`
(exit 0). A config whose hash changed or is gone fails (`GOLDEN MISMATCH`), and so does a
config with output and no blessed hash (`GOLDEN UNBLESSED`), since no check covers it until
it is blessed; each prints the re-record command for its families.

CI runs these from the installed test tree as the CTest entries
`rocke_differential_golden_ll` and `rocke_differential_golden_canonical`, with
`--rocke-shard-from-gtest-env`: each CI shard checks only the families it owns under the
`rocke.tests.ci_shard` rule (the pytest entries' rule), from `GTEST_SHARD_INDEX` and
`GTEST_TOTAL_SHARDS`, so the shards of one run check every family once. Without those
variables (or `ROCKE_TEST_SHARD`) the flag checks every family.

`GOLDEN OK` answers *"does output match the blessed hashes"* and nothing else; it is not the
parity verdict. The byte-identity gate (`tools/check_byte_identity.py`, or `run_diff.py`
without `--python-only`) is what compares the C++ engine with Python.

`--only <family>` checks just the matching families; the rest are not compared and not
reported missing. It matches by substring (`--only conv_implicit_gemm` also selects
`conv_implicit_gemm_dgrad` and `_wgrad`), and a run that selects no family fails.
`--exclude <family>` leaves out the named families (exact names, applied after `--only`).

## Re-bless (only for a reviewed, intended output change)

From the rocke directory, `python tools/golden.py rerecord differential [--only <families>]`
runs both records below (the command a failing check prints). On a merge or rebase conflict
in either golden file, `python tools/golden.py rerecord --conflicted` re-records it on the
merged tree.

```bash
python3 run_diff.py --mode ll --python-only --record-golden --flavor all
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

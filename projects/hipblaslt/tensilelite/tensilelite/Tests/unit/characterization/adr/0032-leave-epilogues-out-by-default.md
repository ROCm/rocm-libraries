# ADR 0032: Leave the epilogue settings out of extracted configs by default

Status:  Accepted
Defect:  none — behavior is intended

## Context
geko writes the epilogue settings into a tuning config only when its
`EPILOGUES` option is on; otherwise its config generator writes them commented
out. Its merge step (`merge_solutions(..., epilogues=True)`, which calls
`Library.add_epilogues`) then sets `Activation`, `ActivationType:
hipblaslt_all`, `UseBias`, `BiasDataTypeList`, `UseScaleAlphaVec` and, for 8-bit
data types, `UseScaleAB: Scalar` on every library it writes, and renames the
kernels to match. Tuning is run without epilogues by convention
(`EPILOGUES: false`), so a library records epilogues its tuning run never had.

ADR 0031 made `TensileLibLogicToYaml` carry every problem type key a library
records, so a config extracted from such a library tunes the shipped epilogue
kernel (`..._Bias_HA_S_SAV_UserArgs_...`) instead of the kernel the run measured
(`..._UserArgs_...`). The converter before ADR 0031 matched neither: it dropped
`ActivationType` and kept `Activation`, `UseBias` and `UseScaleAlphaVec`.

ProblemType reads two of these keys by presence. With `Activation` present,
`ActivationType` is taken from the block whatever `Activation`'s value; with
`UseBias` present, a missing `BiasDataTypeList` is derived from the data types.
Setting the keys to off does not turn the epilogues off.

## Decision
- By default, remove `EPILOGUE_PROBLEM_TYPE_KEYS`, the six keys `add_epilogues`
  writes, from the problem type before the config is formed, and drop the bias
  types a benchmark data header records. `--keep-epilogues`
  (`keepEpilogues=True`) keeps both as recorded. `ActivationHPA`, which geko's
  generator also writes, is not read by ProblemType.
- Name the settings that were left out, meaning those whose removal changes the
  problem type Tensile builds, and point to `--keep-epilogues`.
- A handwritten custom kernel keeps its epilogue settings, with a note. Its
  problem type is fixed by its source, and Tensile exits when a config names a
  custom kernel whose problem type differs; `ActivationType` is the only field
  it takes from the config.
- `formProblemSize` writes `BiasTypeArgs` only when there are bias types.
  Tensile ignores the entry when bias is off and exits on an empty one when bias
  is on, so an empty list never helps. Without bias or gate types,
  `BenchmarkFinalParameters` holds only `ProblemSizes`, as a geko config does.
- Solution parameters are unchanged. `ActivationFuncCall`, which
  `add_epilogues` sets to `False`, is what `Solution.assignDerivedParameters`
  derives whenever there is no activation, so carrying it changes nothing.

Changed expectations: the gfx950 golden in `Tests/unit/test_TensileLibLogicToYaml.py`
loses `Activation`, `UseBias`, `UseScaleAlphaVec` and `BiasTypeArgs` (with
`keepEpilogues` the output is the previous golden byte for byte);
`test_emitted_config_carries_file_defaults_and_target` moves its `ActivationType`
assertion to `test_emitted_config_keeps_epilogues_when_asked`;
`test_benchmark_tree_to_config` no longer carries the header's bias types, which
`test_benchmark_tree_keeps_header_bias_types_with_epilogues` now checks; and the
`main()` stubs take the new argument.

## Consequences
- For each of the five data types of a gfx1250 tuning run with `EPILOGUES:
  false` (bf16, fp8, mxfp8, mxfp4, nvfp4), the config extracted from the merged
  library builds the same problem type as the run's tuning config. For bf16 it
  regenerates the tuned kernel's kernel and solution names, and with
  `--keep-epilogues` the shipped kernel's.
- `test_emitted_config_regenerates_the_same_kernel` runs both ways: kept, the
  config rebuilds the logic's kernel; by default, the kernel of the same logic
  with its epilogue keys removed.
- A library tuned with epilogues on (geko's default when `EPILOGUES` is unset),
  and a benchmark data file from such a run, need `--keep-epilogues` to
  reproduce that run; the note says so.
- Epilogue-like keys geko never adds (`UseE`, `Gradient`, `UseScaleCD`,
  `OutputAmaxD`, `UseGateResidual`) are carried as before: they define a
  different problem, not an option added after tuning.

**Rejected alternatives:**
- Keep the epilogues by default and make leaving them out opt-in — rejected:
  the default should reproduce a tuning run, and tuning runs without epilogues.
- Set the keys to their off values — rejected: ProblemType reads `Activation`
  and `UseBias` by presence (see Context).
- Detect whether a library's epilogues were added after tuning — rejected:
  `add_epilogues` leaves no reliable marker; even `ActivationFuncCall: false` is
  what Tensile derives without activation.

# ADR 0012: Flatten the `BiasTypeArgs` shape written into the solutions header

Status:  Accepted
Defect:  none — behavior is intended

## Context
`LibraryIO._writeSolutionsHeader` emitted

    f.write("- BiasTypeArgs: [{}]\n".format([btype.value for btype in ...]))

Python's `"[{}]".format([7])` yields `"[[7]]"` — the list's own repr already
carries brackets, so the literal pair in the format string wraps it a second
time. Every benchmark data file written since the line was introduced
(3d2048658d9, Dec 2022) therefore carries `- BiasTypeArgs: [[7]]` where the
benchmark *config* schema takes a flat `[7]`. The `GateTypeArgs` line added in
Jul 2026 (c42e983504) is a verbatim copy of the same mistake.

The shape went unnoticed for three years because the only reader of that
header, `LibraryIO.parseSolutionsData`, tests for the *presence* of the
`BiasTypeArgs` key to advance `solutionStartIdxInData` and never binds the
value. Nothing validated it until `TensileLibLogicToYaml` began copying
benchmark data headers into generated configs, where `BiasTypeArgs` *is*
parsed: `SolutionStructs.Solution.BiasTypeArgs` iterates the list and hands
each element to `DataType`, which rejects a list with
`RuntimeError: initializing DataType to <class 'list'> [7]`.

Evidence that flat is the intended shape, not a second schema:
- The sibling `Exact` line two lines above interpolates a list with no added
  brackets, in the same function and the same commit.
- `ActivationArgs`, written by the same function, is emitted in exactly the
  shape its benchmark-config consumer walks — so the header's convention is to
  emit consumer-ready values. Bias and Gate are one level deeper than that.
- Every other producer of the field is flat: `tensile_config_generator.py`,
  `ClientWriter.py`, geko's config generator, `TensileLibLogicToYaml`'s own
  library-logic path, and every hand-authored config in the tree.

## Decision
Fix the writer (drop the literal brackets from both the `BiasTypeArgs` and
`GateTypeArgs` lines) **and** normalize on read in `TensileLibLogicToYaml`,
because the writer fix does not retroactively repair the benchmark data files
already on disk — those keep the nested shape and would keep crashing.
`normalizeBiasTypeArgs` flattens one level and maps an empty result to `None`
so the `BiasDataTypeList` fallback engages (the old writer emitted `[[]]` for
an empty bias list, which is truthy and would otherwise be read as one bias
type that is itself an empty list).

Re-record the one affected golden, `test_header_with_bias_and_activation` in
`LibraryIO/__snapshots__/test_writesolutions_char.ambr` — the only snapshot in
the suite that exercises a header with bias set. It pinned the buggy output,
which is what a characterization golden is for; ADR 0010 set the same precedent
in reverse ("when that lands, flip this golden").

## Consequences
- Newly written benchmark data files carry `- BiasTypeArgs: [7]`. Old files
  keep `[[7]]`; both now read correctly, so the corpus can stay mixed.
- No runtime reader breaks: `parseSolutionsData` tests key presence only, and
  the key is still written in both shapes, so `solutionStartIdxInData` is
  unchanged. `_findBodyOffset` is likewise key-based.
- The header is two bytes shorter per affected line (four when both
  `BiasTypeArgs` and `GateTypeArgs` are written), so the first refresh of a
  stale cached solutions file takes the body-rewrite branch in `writeSolutions`
  rather than the in-place branch. Output is equivalent; cost is one extra
  rewrite.
- Only `BiasTypeArgs` is normalized on read. Nothing reads `GateTypeArgs` out of
  a benchmark data header today — `formProblemSize` re-derives it from the
  problem type's `GateResidualDataTypeList` — so the gate change is a writer-side
  fix only, pinned by `test_header_with_bias_and_gate`.

**Rejected alternatives:**
- Fix only the writer — rejected: leaves every already-generated file crashing
  through `TensileLibLogicToYaml`, which is how the defect was found.
- Normalize only in the converter — rejected: leaves the writer emitting a
  shape no consumer accepts, so the next tool to read the field hits it again.
- Treat the nested form as the solutions-yaml schema and convert on copy —
  rejected: no reader anywhere accepts nesting, and the `ActivationArgs`
  control in the same function shows the intended convention.

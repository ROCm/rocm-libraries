# Correct and refresh the post-mutation coverage baseline

## Status

Accepted

## Context

ADR 0018 says the previous rebaseline lowered nine per-file coverage floors.
The committed diff lowers ten. It also names `KernelWriterAssembly.py`, whose
floor rose from 81.88% to 83.66%, while omitting two actual reductions:
`Configuration.py` fell from 99.25% to 92.53%, and `Solution.py` fell from
76.77% to 73.02%.

A source comparison from the recorded rebase commit shows that seven of the ten
lowered files were unchanged: `AsmAddressCalculation.py`, `Component.py`,
`GlobalWriteBatch.py`, `StreamK.py`, `Configuration.py`,
`KernelWriterModules.py`, and `segment_interleave.py`. `GSU.py`, `Solution.py`,
and `TensileCreateLibrary/Run.py` changed. This supersedes ADR 0018's count and
classification; it does not alter those already-recorded floor values.

The follow-up validation removes three configurations that reached unrelated
emitter code only before an expected MX local-read exception. A current
merge-tree coverage run therefore measures `LocalRead.py` at 76.68% and
`LraTileAssignment.py` at 93.17%. Both decreases are consequences of removing
that invalid-path coverage. The same run and the public coverage artifact both
measure unchanged `PackData.py` at 85.04%, below its stale 89.06% floor.

## Decision

Retain the ten original lowered values and correct their evidence here. Lower
only the three newly reproduced floors: `LocalRead.py`,
`LraTileAssignment.py`, and `PackData.py`. Raise ten floors reached by the
current tests and add the two newly measured source files. Keep the tolerance
at one percentage point.

## Consequences

The committed baseline matches the current source and records every reduction
instead of silently weakening `Configuration.py` and `Solution.py`. Removing
the exception-path tests no longer leaves the coverage gate permanently red.
Future drops beyond the existing tolerance still require an explicit,
file-specific update.

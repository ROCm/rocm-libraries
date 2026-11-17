# ADR 0035: Name split-GSU kernels by their original numeric GSU

Status:  Accepted
Defect:  AIHPBLAS-4297

## Context

ADR 0003 pinned a `TypeError` in `getKernelNameMin`: the split-GSU path changed
GSU values greater than one and automatic GSU (`-1`) to the string `"M"`, then
compared that string with zero. Split-GSU naming needs a distinct kernel name
for the unit and multi-GSU variants. The characterization tests deliberately
expected the crash until a specification test could replace that expectation.

## Decision

Decide whether to include `GlobalSplitU` in a kernel name from the saved
numeric value, before it is masked. In split mode, leave the GSU 1 name
unchanged and retain the `GSUM` token for values greater than one and `-1`.
Unsplit naming retains its existing treatment of GSU values.

## Consequences

`getKernelNameMin` and its file-name callers can name split and automatic GSU
kernels without the exception. GSU 2 and automatic GSU share the `GSUM` kernel
name, while GSU 1 remains distinct. The previous TypeError characterizations
now assert the intended names. The mutation equivalents recorded for the crash
in `DECISIONS.md` require reevaluation against this observable behavior.

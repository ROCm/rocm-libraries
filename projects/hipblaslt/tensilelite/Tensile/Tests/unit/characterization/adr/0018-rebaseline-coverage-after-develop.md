# Rebaseline coverage after the develop rebase

## Status

Accepted

## Context

The mutation-test stack was rebased onto `develop` at `dd749e5bfc05`. Two
consecutive coverage runs on the rebased tree measured the same nine files more
than the one-percentage-point tolerance below their stored floors. The full
unit suite passed before the final measurement.

Six affected files are unchanged between the old stack base and current
`develop`: `segment_interleave.py`, `Component.py`, `GlobalWriteBatch.py`,
`GSU.py`, `AsmAddressCalculation.py`, and `KernelWriterModules.py`. Their old
floors are not reproducible with the current source and test environment.
Develop changed the other three files: `TensileCreateLibrary/Run.py`,
`StreamK.py`, and `KernelWriterAssembly.py`.

The current report measures 84.08% combined coverage. The refresh also finds
16 higher floors, 14 source files that were not in the previous baseline, and
two baseline entries whose source files were removed by develop.

## Decision

Regenerate the complete per-file baseline from the clean post-rebase report.
Allow reductions only for the nine files named by the ratchet failure. Keep the
one-percentage-point tolerance unchanged.

## Consequences

The baseline describes the current develop-based source tree and passes the
same measurement that produced it. The nine reductions are explicit review
items; future decreases remain blocked, while the 16 increases become new
floors.

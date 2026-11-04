# Common utilities

rocPRIM's tests and benchmarks employ numerous utilities that are common in implementation. This folder hosts these for an easier and less error-prone maintenance.

## When to add a common utility

When adding a new test or benchmark that depends on a utility, the following cases must be considered:

1. If the utility is already implemented in some `common` header, then there's nothing to do except perhaps extending its functionality.
2. If the utility does not exit yet in any `common` header, then fisrt it must be checked whether some `benchmark` or `test`[^1] utility header implements this functionality. If so, then it must be moved to the appropriate common header.
3. If the utility does not exit yet in any `common` nor `test` nor `benchmark` utility header, then it must be added to the appropriate `test` or `benchmark` header.

[^1]: When adding a new test check the `benchmark` utilities, and viceversa.

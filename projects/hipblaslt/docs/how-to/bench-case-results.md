# Validate bench cases

`hipblaslt-bench` can validate GEMMs without collecting timing samples. Set
`iters: 0` and `cold_iters: 0` with `adaptive: false` in each input record, and
enable `norm_check`. Numerical validation still launches the selected solution;
the fixed-count timing loop launches nothing. Ordinary positive iteration counts
and adaptive timing retain their existing behavior.

For data files, `--host_side_fill_kernel` initializes inputs on the host for
every record. This also avoids device initialization kernels during emulation.
The flag applies to the whole process, including all cases in the data file.

## Continue after a numerical failure

Pass `--continue-on-failure` to execute later data-file cases after a case fails
numerical validation. The process still returns failure if any case fails.
Without this flag, a failed case stops the data-file loop. An incomplete case or
runtime exception stops execution even with the flag enabled.

For example, after converting a YAML input into a bench data file:

```sh
hipblaslt-bench --host_side_fill_kernel --continue-on-failure --data cases.data
```

## Read case results

Bench writes line-oriented records prefixed with `[hipblaslt-case-v1]`. They
appear alongside ordinary diagnostics and timing output. Case numbers are
zero-based and count records that reach execution; filtered records do not
receive a number.

| Record | Fields after the record name |
| --- | --- |
| `BEGIN` | case number, requested solution index, first M, first N, batch count, first K |
| `RESULT` | selected solution index, cumulative numerical status |
| `END` | case number, requested solution index, final case status |
| `DONE` | executed case count, failed or incomplete case count |

`BEGIN` precedes the matmul call. `RESULT` is emitted with solution output;
enable `print_solution_found` and `print_kernel_info` to report actual selected
indices. A case that searches several candidates can have several `RESULT`
records. `END` follows a returned matmul call. `DONE` marks the end of a data-file
run, including one stopped by a failed case; a single command-line case has no
`DONE` record.

The numerical statuses are `PASSED`, `FAILED`, and `NOT_CHECKED`. A numerical
failure remains recorded even if a later check passes. The final status can
also be `INCOMPLETE` when execution returned before completing the case.
`NOT_CHECKED` means no norm verdict was recorded; it does not prove numerical
correctness. Norm, allclose, and runtime diagnostics should be read together
with the process exit status.

An explicit-index run with a failed case followed by a passing case can emit:

```text
[hipblaslt-case-v1] BEGIN 0 0 16 16 1 256
[hipblaslt-case-v1] RESULT 0 FAILED
[hipblaslt-case-v1] END 0 0 FAILED
[hipblaslt-case-v1] BEGIN 1 1 16 16 1 512
[hipblaslt-case-v1] RESULT 1 PASSED
[hipblaslt-case-v1] END 1 1 PASSED
[hipblaslt-case-v1] DONE 2 1
```

That process exits with status 1. A missing `END` or `DONE`, an incomplete case,
or a nonzero exit must not be interpreted as successful validation. The records
describe native case execution and numerical checks; race detection requires
the separate detector output.

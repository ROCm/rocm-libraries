# gfx1250 LLVM ISA feature validation

These standalone examples validate each new bridge operation twice: required
intrinsics must appear in generated LLVM text, and required instructions must
appear in `llvm-objdump` output from the compiled code object. The scripts select
the `llvm23` ROCKE flavor.

From `rocke/platform`:

```bash
export PYTHONPATH="$PWD/python"
export ROCM_PATH=/path/to/rocm

python -m rocke.examples.gfx1250.isa_features.scalar_controls_verify
python -m rocke.examples.gfx1250.isa_features.barrier_verify
python -m rocke.examples.gfx1250.isa_features.async_store_verify
python -m rocke.examples.gfx1250.isa_features.global_transpose_verify
python -m rocke.examples.gfx1250.isa_features.tdm_verify
```

Each file can also be run directly with the same `PYTHONPATH`. Use `--arch
gfx1250` to select the target explicitly. Use `--compile-only` when no matching
GPU is available:

```bash
python -m rocke.examples.gfx1250.isa_features.async_store_verify --compile-only
```

`llvm-objdump` is mandatory, because successful compilation alone is not an ISA
check. It is resolved from `LLVM_OBJDUMP`, then `$ROCM_PATH/llvm/bin`, then
`PATH`.

## Coverage

- `scalar_controls_verify.py`: LLVM/ISA and functional. Executes
  `s_delay_alu`, `s_wait_alu`, `s_clause`, and `s_wait_xcnt` in an exact i32
  copy/transform check.
- `barrier_verify.py`: LLVM/ISA and functional for the non-named workgroup split
  barrier. Two wave32 waves exchange LDS values after
  `s_barrier_signal -1`/`s_barrier_wait -1`; LDS traffic is explicitly drained
  before signal. Named-barrier `init`, `signal_var`, `join`, and `wakeup`, plus
  `barrier_leave`, are compile/ISA-only. ROCKE exposes the low-level operations
  but not enough named lifecycle/member-count semantics to launch that probe
  without risking a deadlock.
- `async_store_verify.py`: LLVM/ISA and functional. Each lane stages a
  deterministic 16-byte record in LDS, issues 1-, 4-, 8-, and 16-byte
  asynchronous LDS-to-global stores to disjoint output regions, drains
  `ASYNCcnt`, and compares every output byte.
- `global_transpose_verify.py`: mandatory LLVM/ISA checks for f16, bf16, and i16.
  Functional execution is intentionally skipped because the exposed ROCKE API
  does not document the exact wave32 lane permutation needed to construct a
  trustworthy host reference.
- `tdm_verify.py`: mandatory LLVM/ISA checks for tensor load, tensor store, and
  `tensorcnt`. Functional execution is intentionally skipped. Descriptor groups
  are zero only to validate code generation; they are never launched because
  ROCKE does not yet expose construction of a valid D# global-memory/LDS
  descriptor.

`--compile-only` additionally skips the three otherwise-safe functional checks.
An intentional functional skip is reported as `SKIP` and does not hide a failed
LLVM or ISA check.

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
python -m rocke.examples.gfx1250.isa_features.data_prefetch_verify
python -m rocke.examples.gfx1250.isa_features.cluster_ids_verify
python -m rocke.examples.gfx1250.isa_features.multicast_verify
python -m rocke.examples.gfx1250.isa_features.prefetch_users_verify
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
- `tdm_verify.py`: LLVM/ISA and functional. Descriptors come from
  `IRBuilder.tdm_descriptor_2d`. Each workgroup of a 2x2 grid moves one 16x64
  i32 tile of a 32x128 tensor, stored with a row pitch of 160, from global to
  LDS with `tensor_load_to_lds`, then back to global with
  `tensor_store_from_lds`. It drains `TENSORcnt` after each transfer. Each lane
  also reads its LDS words directly, which checks the row-major LDS layout
  independently of the store. LLVM must show the descriptor built from the
  pointer (`ptrtoint`, `readfirstlane`, and the 4-byte element flag word). ISA
  must match both transfers, `s_wait_tensorcnt 0x0`, and the LDS read-back. The
  functional check compares the LDS read-back and the output exactly, and
  checks that the pitch gap is left untouched.
- `data_prefetch_verify.py`: LLVM/ISA and functional. Enables scalar prefetch
  through `s_setreg` on `MODE`, then issues `s_prefetch_data` through global,
  constant, and flat views of one buffer, `s_buffer_prefetch_data` through a
  buffer resource, `global_prefetch_b8`, and `flat_prefetch_b8` before an exact
  i32 copy/transform. Prefetch is a hint, so the check is that results are
  unchanged.
- `cluster_ids_verify.py`: mandatory LLVM/ISA checks for every workgroup-cluster
  read and the cluster barrier. The per-axis ids lower to `ttmp9`/`ttmp7` and
  4-bit `ttmp6` field extracts, the flat id to `s_getreg_b32` of
  `HW_REG_IB_STS2`, and `cluster_barrier` to the workgroup barrier followed by
  a first-wave `s_barrier_signal -3` and `s_barrier_wait -3` (disassembled as
  `0xfffd`), bracketed by the
  cluster-scope release (`s_wait_storecnt`) and acquire (`global_inv
  scope:SCOPE_SE`). A second build declares `cluster_dims` of 2x2x1 and must
  also carry `"amdgpu-cluster-dims"="2,2,1"` in LLVM and `.cluster_dims` in the
  code-object metadata (read with `llvm-readelf --notes`, resolved like
  `llvm-objdump` from `LLVM_READELF`). With the shape compiled in, the backend
  folds the per-axis max ids and `cluster_size` to constants, so that build
  does not require their `ttmp6` extracts. Functionally, every workgroup records
  each read: a plain launch on a 4x4x1 grid must behave as 1x1x1 clusters, and
  a launch with `cluster=(2, 2, 1)` on the same grid must report the cluster
  and in-cluster ids the host reference derives from the launch shape.

- `multicast_verify.py`: LLVM/ISA and functional for the cluster multicast
  loads. One kernel with a `cluster_dims` of 4x1x1 issues `cluster_load_b32`,
  `_b64`, and `_b128`, then `cluster_load_async_to_lds_b8`, `_b32`, `_b64`, and
  `_b128` into LDS. It waits with `s_wait_asynccnt 0x0` before reading LDS back.
  The participation mask travels in `M0`. The backend may compute it there
  directly (for example `s_lshl_b32 m0, ...`), so the ISA check accepts any
  scalar write to `M0`. The functional runs launch an 8x1x1 grid with
  `cluster=(4, 1, 1)` and multicast groups of 1, 2, and 4 workgroups. Every
  member of a group loads the record of the group's first workgroup, so data
  from a neighbouring group is caught.
- `prefetch_users_verify.py`: flag off against flag on for
  `BlockScaledGemmSpec(prefetch=True)`, on the `wmma`, `wmma_scale`, and
  `wmma_scale16` paths.
  - The llvm23 backend already emits one `global_prefetch_b8` and a
    `HW_REG_WAVE_MODE` bit 25 `s_setreg` in every gfx1250 prologue, so the
    probe compares instruction counts.
  - The flag must add exactly one `hwreg(HW_REG_WAVE_MODE, 24, 1)` write (the
    scalar-prefetch enable), at least one `s_prefetch_data`, and more
    `global_prefetch_b8` instructions.
  - The `v_wmma` count must stay the same.
  - Functionally, both builds must match the GEMM reference and each other bit
    for bit. `gemm/block_scaled_gemm_verify.py --prefetch` runs the full GEMM
    verifier with the flag on.

`--compile-only` additionally skips every otherwise-safe functional check.
An intentional functional skip is reported as `SKIP` and does not hide a failed
LLVM or ISA check.

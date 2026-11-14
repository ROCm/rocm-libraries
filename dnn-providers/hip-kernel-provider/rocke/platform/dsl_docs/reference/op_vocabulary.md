# IR Op Vocabulary

Complete reference of operations recognized by `core/ir.py` and lowered to AMDGPU LLVM IR. The native Python lowerer (`core/lower_llvm.py`) and the C++ engine (`cpp/`, reached through the `rocke_engine` extension) are two interchangeable engines that emit byte-identical IR for these ops; `core/backend.py` selects between them. Verified against the IR source. Each entry lists the operation name (as it appears in `print_ir`), the IRBuilder method, and the LLVM emission summary.

## Arithmetic

| Op name                | Builder                           | LLVM emission                          |
|------------------------|-----------------------------------|----------------------------------------|
| `arith.constant`       | `const_i32`, `const_i64`, `const_f32`, `const_f64`, `fp16_zero` | constant value |
| `arith.constant_vec`   | `zero_vec_f32`, `zero_vec`        | `zeroinitializer` vector               |
| `arith.add`            | `add`, `add_nuw`                  | `add nsw` (`add`), `add nuw` (`add_nuw`, attr `wrap = "nuw"`) |
| `arith.sub`            | `sub`                             | `sub`                                  |
| `arith.mul`            | `mul`                             | `mul`                                  |
| `arith.div`            | `div`                             | `sdiv`                                 |
| `arith.mod`            | `mod`                             | `srem`                                 |
| `arith.shl`            | (internal, used by `warp_shuffle_xor`) | `shl`                              |
| `arith.fadd`           | `fadd`                            | `fadd`                                 |
| `arith.fsub`           | `fsub`                            | `fsub`                                 |
| `arith.fmul`           | `fmul`                            | `fmul`                                 |
| `arith.fdiv`           | `fdiv`                            | `fdiv`                                 |
| `arith.fneg`           | `fneg`                            | `fneg`                                 |
| `arith.fmax`           | `fmax`                            | `llvm.maxnum.{f64,f32,f16,bf16}`       |
| `arith.fmin`           | `fmin`                            | `llvm.minnum.{f64,f32,f16,bf16}`       |
| `arith.fdot2`          | `fdot2`                           | `llvm.amdgcn.fdot2` (f16) / `llvm.amdgcn.fdot2.f32.bf16` (bf16), clamp off; f32 accumulator |
| `arith.cmp`            | `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`, `cmp_ult` | `icmp {slt,sle,sgt,sge,eq,ne,ult}` |
| `arith.assume_range`   | (private `_assume_range`, for checked callers only; written by `gpu_checked_value`) | nothing: uses of the result print the marked value; the attrs (`ge` or a `ge` operand plus `ge_plus`; `lt` or an `lt` operand plus `lt_plus`; `multiple_of`, `residue`; `by`, the writer) record a range and congruence for `stated_facts_of` and the tiling layer's range walk |
| `arith.fcmp`           | `fcmp(pred, ...)`                 | `fcmp {olt,ole,ogt,oge,oeq,one,ord,uno}` |
| `arith.and`            | `land`                            | `and`                                  |
| `arith.or`             | `lor`                             | `or`                                   |
| `arith.xor`            | (internal)                        | `xor`                                  |
| `arith.not`            | `lnot`                            | `xor` against all-ones                 |
| `arith.zext`           | `zext`                            | `zext`                                 |
| `arith.sext`           | `sext`                            | `sext`                                 |
| `arith.select`         | `select`, `masked_select`         | `select`                               |
| `arith.bitcast`        | `bitcast`                         | `bitcast`                              |
| `arith.cast_to_f32`    | `cast_to_f32`                     | `fpext` (f16/bf16 -> f32)              |
| `arith.cast_f32_to`    | `cast_f32_to`                     | `fptrunc` to f16/bf16                  |
| `arith.trunc_f32_to_f16` | `trunc_f32_to_f16`              | `fptrunc` (vec-packed in some emissions) |
| `arith.sitofp_f32`     | `sitofp_f32`                      | `sitofp i32 to float`                  |
| `arith.cvt_f32_to_f64` | `cvt_f32_to_f64`                  | `fpext float to double` (exact)        |
| `arith.cvt_f64_to_f32` | `cvt_f64_to_f32`                  | `fptrunc double to float` (round to nearest even) |
| `arith.cvt_fp8_to_f32` | `cvt_fp8_to_f32`                  | `llvm.amdgcn.cvt.f32.fp8`              |
| `arith.cvt_bf8_to_f32` | `cvt_bf8_to_f32`                  | `llvm.amdgcn.cvt.f32.bf8`              |
| `arith.cvt_f32_to_fp8` | `cvt_f32_to_fp8`                  | `llvm.amdgcn.cvt.pk.fp8.f32` (low byte) |
| `arith.cvt_f32_to_bf8` | `cvt_f32_to_bf8`                  | `llvm.amdgcn.cvt.pk.bf8.f32` (low byte) |
| `arith.cvt_f32_to_i8_sat` | `cvt_f32_to_i8_sat`            | round-to-nearest + saturating cast     |

## Math

| Op name        | Builder      | LLVM intrinsic                  |
|----------------|--------------|---------------------------------|
| `math.exp2`    | `exp2`       | `llvm.exp2.f32`                 |
| `math.sqrt`    | `sqrt`       | `llvm.sqrt.f32`                 |
| `math.rsqrt`   | `rsqrt`      | `llvm.amdgcn.rsq.f32`           |
| `math.tanh`    | `tanh`       | piecewise f32 expansion: OCML polynomial for `abs(x) < 0.625`, otherwise `exp2` + reciprocal; bitwise sign restoration |
| `math.rcp`     | `rcp`        | `1.0 / v` (hardware reciprocal) |

`exp2`, `log2`, `sqrt`, `rsqrt`, `rcp_fast`, `fma`, `fmax3`, `fmin3` and `vector_fma` have no f64 lowering and reject f64 inputs in both the LLVM and HIP lowerings. `exp2_fast` rejects f64 in the LLVM lowering and has no HIP lowering for any dtype. `tanh` accepts only f32: the builder rejects any other operand type. `rcp` accepts f64, and `fabs` accepts f64 (`llvm.fabs.f64`, HIP `fabs`).

`clamp_f32(v, lo, hi)` is `fmin(hi, fmax(lo, v))` — folds to `v_med3_f32`.

## Vector

| Op name              | Builder                  | LLVM emission                  |
|----------------------|--------------------------|--------------------------------|
| `vector.add`         | `vector_add`             | elementwise vector `fadd`/`add` |
| `vector.mul`         | `vector_mul`             | elementwise vector multiply     |
| `vector.max`         | `vector_max`             | elementwise vector max          |
| `vector.sum`         | `vector_sum`             | `llvm.vector.reduce.fadd.*`     |
| `vector.reduce_max`  | `vector_reduce_max`      | `llvm.vector.reduce.fmax.*`     |
| `vector.splat`       | `vector_splat`           | `insertelement` + `shufflevector` |
| `vector.select`      | `vector_select`          | vector `select`                 |
| `vector.bitcast`     | `vec_bitcast`            | `bitcast` between equal-size vectors |

## GPU / Lane

| Op name             | Builder            | LLVM emission                        |
|---------------------|--------------------|--------------------------------------|
| `gpu.thread_id`     | `thread_id_x`      | `llvm.amdgcn.workitem.id.x()`        |
| `gpu.block_id`      | `block_id_x/y/z`   | `llvm.amdgcn.workgroup.id.{x,y,z}()` |
| `tile.lane_id`      | `lane_id`          | `mbcnt.hi(-1, mbcnt.lo(-1, 0))`      |
| `tile.readfirstlane`| `readfirstlane`    | `llvm.amdgcn.readfirstlane.{i32,i64}` |
| `tile.pin_sgpr`     | `pin_sgpr`         | `asm volatile("" : "+s"(x))`         |
| `tile.wave_all`     | `wave_all`         | ballot + compare                     |
| `tile.wave_any`     | `wave_any`         | ballot + compare                     |
| `tile.wave_ballot`  | `wave_ballot`      | `llvm.amdgcn.ballot.i64`             |
| `tile.ds_bpermute`  | `ds_bpermute`      | `llvm.amdgcn.ds.bpermute`            |

`warp_shuffle_xor(v, lane_xor)` is a helper that combines `xor` + `ds_bpermute` and bitcasts as needed.

## Memory

### Global

| Op name                       | Builder                       | LLVM emission |
|-------------------------------|-------------------------------|---------------|
| `memref.global_load`          | `global_load_f16`             | typed `load addrspace(1)` |
| `memref.global_load_typed`    | `global_load`, `global_load_i8`, `global_load_i16`, `global_load_i32`, `global_load_i64`, `global_load_fp8e4m3`, `global_load_bf8e5m2`, `global_load_bf16`, `global_load_f32`, `global_load_f64` | typed `load addrspace(1)` |
| `memref.global_load_vN`       | `global_load_vN`, `global_load_vN_f16` | vector `load addrspace(1)` (n in {2,4,8}) |
| `memref.global_store_typed`   | `global_store`                | typed `store addrspace(1)` |
| `memref.global_store_vN`      | `global_store_vN`             | vector `store addrspace(1)` |
| `memref.global_atomic_add_f32`| `global_atomic_add_f32`       | `atomicrmw fadd addrspace(1)` |
| `memref.fence`               | `fence`                       | `fence syncscope(...) <ordering>` |
| `memref.global_flag_store`   | `global_flag_store`           | `store atomic i32 ... syncscope(...) <ordering>` |
| `memref.global_flag_wait_eq` | `global_flag_wait_eq`         | spin loop around `load atomic i32 ... acquire` |

`masked_global_load(ptr, idx, mask, other, dtype)` clamps the false-lane index to 0 before loading, then selects `other` afterward.

A pointer parameter declared `addr_space="constant"` is `ptr addrspace(4)` in the kernel header. The
three global loads above read through it as `load addrspace(4)`; they refuse a flat or LDS pointer.
Every other op that addresses global memory through it (the stores, atomics and flag ops above,
`cooperative_global_store`, `global_prefetch`, `global_load_tr16_b128`, the cluster loads, and the
copies between global memory and LDS) is refused by the LLVM lowering of both engines. Each of these
refusals is a `TypeError` (`ROCKE_ERR_TYPE` in the C++ engine) that names the op's builder.

### Global pointers

| Op name                  | Builder             | LLVM emission                                        | HIP emission |
|--------------------------|---------------------|------------------------------------------------------|--------------|
| `tile.global_ptr_add`    | `global_ptr_add`    | `getelementptr inbounds i8, ptr addrspace(1)`        | `(const char*)p + (int64_t)off` |
| `tile.global_ptr_offset` | `global_ptr_offset` | `getelementptr i8, ptr addrspace(1)` (no `inbounds`) | `(T*)((uintptr_t)p + (uint64_t)off)` |
| `tile.global_ptr_to_i64` | `global_ptr_to_i64` | `ptrtoint ptr addrspace(1) ... to i64`               | `(int64_t)(uintptr_t)p` |

`global_ptr_add` promises the compiler that the result stays inside the allocation `p` points into;
`global_ptr_offset` does not, so its result may point outside it (before it, with a negative offset)
-- for an address that is only compared, converted to an integer, or given to a buffer descriptor
that range-checks every access. Its offset is i64 and is never widened implicitly; widen an i32 with
`sext` or `zext`. The HIP result keeps `p`'s element type `T`, so indexing it counts elements.

### LDS

| Op name                | Builder                  | LLVM emission                                  |
|------------------------|--------------------------|------------------------------------------------|
| `tile.smem_alloc`      | `smem_alloc`             | module-level `addrspace(3) global`             |
| `tile.smem_addr_of`    | `smem_addr_of`           | `ptrtoint` of LDS base                         |
| `tile.smem_ptr_add`    | `smem_ptr_add`           | `add i64` over LDS addresses                   |
| `tile.smem_load`       | `smem_load_f16`          | scalar `load addrspace(3)`                     |
| `tile.smem_load_v4`    | `smem_load_v4_f16`       | `<4 x half>` load                              |
| `tile.smem_load_vN`    | `smem_load_vN_f16`, `smem_load_vN` | `<N x dtype>` load (n in {1,2,4,8})  |
| `tile.smem_store`      | `smem_store_f16`         | scalar `store addrspace(3)`                    |
| `tile.smem_store_vN`   | `smem_store_vN_f16`, `smem_store_vN` | `<N x dtype>` store                |
| `tile.ds_read_tr16_b64`| `ds_read_tr16_b64`       | `llvm.amdgcn.ds.read.tr16.b64`                 |

### Buffer (addrspace(8))

| Op name                          | Builder                          | LLVM emission                                                |
|----------------------------------|----------------------------------|--------------------------------------------------------------|
| `tile.buffer_rsrc`               | `buffer_rsrc`                    | `llvm.amdgcn.make.buffer.rsrc.p1(ptr, 0, num_bytes, 159744)` |
| `tile.buffer_load_f16`           | `buffer_load_f16`                | `raw.ptr.buffer.load.i32` + truncate / bitcast               |
| `tile.buffer_load_vN_f16`        | `buffer_load_vN_f16`             | `raw.ptr.buffer.load.{i32,v2i32,v4i32}` + `vec_bitcast`      |
| `tile.buffer_store_f16`          | `buffer_store_f16`               | `raw.ptr.buffer.store.i16`                                   |
| `tile.buffer_store_vN_f16`       | `buffer_store_vN_f16`            | `raw.ptr.buffer.store.{i32,v2i32,v4i32}`                     |
| `tile.buffer_load_vN`            | `buffer_load_vN`                 | `raw.ptr.buffer.load.{i8,i16,i32,v2i32,v4i32}` + `bitcast` to `<n x dtype>` |
| `tile.buffer_store_vN`           | `buffer_store_vN`                | `bitcast` + `raw.ptr.buffer.store.{i8,i16,i32,v2i32,v4i32}`  |
| `tile.async_buffer_load_lds`     | `async_buffer_load_lds`          | `llvm.amdgcn.raw.ptr.buffer.load.lds`                        |
| `tile.async_buffer_load_lds_addr`| `async_buffer_load_lds_addr`     | same as above; explicit LDS i64 address                       |
| `tile.buffer_atomic_fadd_f32`    | `buffer_atomic_fadd_f32`         | `raw.ptr.buffer.atomic.fadd.f32` (pre-add result unused)     |

`buffer_load_vN` / `buffer_store_vN` take any scalar whose width is whole bytes (i8, u8, fp8, bf8,
i16, f16, bf16, i32, f32, i64, f64); one access moves 1, 2, 4, 8 or 16 bytes. The hardware checks an
access of 4 bytes or more per 4-byte component and a 1- or 2-byte access whole: a load past
`num_records` reads 0, a store past it writes nothing. The HIP lowering has no vector type for u8 or
i64 elements, so those two lower through LLVM only.

`buffer_atomic_fadd_f32(rsrc, voffset, soffset, value, cachepolicy=0)` adds an f32 into global
memory through a buffer resource; `voffset` and `soffset` are byte offsets. The returned pre-add value
is discarded, so the backend selects the no-return atomic. Guard an invalid offset with `scf_if`
instead of steering it out of range: a load at the out-of-range sentinel reads 0, but an atomic there
faults on gfx1250. `cachepolicy` is the intrinsic's raw immediate, an int in 0..31 that both engines
check in the builder and again in the lowering. On gfx12 and later its bits [2:0] are `th` and [4:3]
`scope`. `rocke.core.ir.CPOL_ATOMIC_CASCADE_DEV` (C++: `ROCKE_CPOL_ATOMIC_CASCADE_DEV`) is
`th = TH_ATOMIC_CASCADE (4)` with `scope = SCOPE_DEV (2)`, i.e. 20. With it, the WGP and L2 caches add
locally and write the sum back once, which removes most of the global read-modify-write traffic of a
cross-workgroup reduction. The destination must be zeroed first. The encoding is gfx12+, so callers
pass 0 on gfx9 and gfx11. The op has no HIP lowering.

`buffer_rsrc` takes a global pointer: the intrinsic's overload names `ptr addrspace(1)`, so a
parameter declared `addr_space="constant"` (`ptr addrspace(4)` in the kernel header) is refused by
the LLVM lowering of both engines, as the `global_ptr_*` ops refuse it. `num_bytes` is i32 or i64 at
llvm22 and later (an i32 is zero-extended); llvm20's intrinsic takes an i32 `num_records`, so an i64
`num_bytes` is refused there. The HIP lowering passes `num_records` to
`__builtin_amdgcn_make_buffer_rsrc` cast to `(int)`, which would truncate a wider byte count, so it
refuses any `num_bytes` but i32. It keeps a constant-space pointer, since the HIP signature declares
every pointer parameter alike and the builtin takes a `void*`.

## MFMA

| Op name                          | Operands (per-lane)              | Output (per-lane) |
|----------------------------------|----------------------------------|--------------------|
| `tile.mfma_f32_16x16x16_f16`     | `<4xhalf>`, `<4xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_16x16x32_f16`     | `<8xhalf>`, `<8xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_16x16x16_bf16`    | `<4xi16>` (bf16 bitcast), same, `<4xfloat>` | `<4xfloat>` |
| `tile.mfma_f32_16x16x32_bf16`    | `<8xbfloat>`, same, `<4xfloat>`  | `<4xfloat>`        |
| `tile.mfma_f32_32x32x8_f16`      | `<4xhalf>`, `<4xhalf>`, `<16xfloat>` | `<16xfloat>`   |
| `tile.mfma_f32_32x32x16_f16`     | `<8xhalf>`, `<8xhalf>`, `<16xfloat>` | `<16xfloat>`   |
| `tile.mfma_f32_4x4x4_f16`        | `<4xhalf>`, `<4xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_4x4x4_bf16`       | `<4xi16>` (bf16 bitcast), same, `<4xfloat>` | `<4xfloat>` |

All MFMAs take three literal immarg constants (`cbsz=0, abid=0, blgp=0`) at the LLVM level.

## Synchronization / Scheduling

| Op name                  | Builder                | LLVM emission                                                |
|--------------------------|------------------------|--------------------------------------------------------------|
| `tile.s_waitcnt`         | `s_waitcnt`            | `llvm.amdgcn.s.waitcnt(i32 encoded_imm)`                     |
| `tile.sync`              | `sync`                 | `s_waitcnt(vmcnt=0, lgkmcnt=0)` + `s.barrier`                |
| `tile.sync_lds_only`     | `sync_lds_only`        | `s_waitcnt(lgkmcnt=0)` + `s.barrier`                         |
| `tile.sync_half_block`   | `sync_half_block`      | conditional barrier (correctness depends on caller control flow) |
| `tile.sched_group_barrier` | `sched_group_barrier` | `llvm.amdgcn.sched.group.barrier(mask, count, sync_id)`      |
| `tile.sched_barrier`     | `sched_barrier`        | `llvm.amdgcn.sched.barrier(mask)`                            |
| `tile.s_setprio`         | `s_setprio`            | `llvm.amdgcn.s.setprio(i16 prio)`                            |

`s_waitcnt` immarg encoding on gfx950 spans both `[3:0]` and `[15:14]` for vmcnt; LGKM is 4 bits and clamps at 15.

## Control Flow

| Op name      | Builder                | Lowering                                              |
|--------------|------------------------|-------------------------------------------------------|
| `scf.for`    | `scf_for`              | header / body / latch / exit blocks + phi (induction) |
| `scf.for_iter` | `scf_for_iter`       | header / body / latch / exit + phi for each iter_var  |
| `scf.if`     | `scf_if` (ctx manager) | conditional branch + then block + join block          |
| `scf.if_else` | `scf_if_else(cond, result_types=())` (ctx manager) | conditional branch + then and else blocks + join block; one phi per result |
| `scf.yield`  | `scf_yield(*values)`   | records yielded values; consumed by enclosing for/if_else |
| `cf.return`  | (auto-emitted)         | `ret void`                                            |
| `tile.trap`  | `trap()`               | `call void @llvm.trap()` (`s_trap 2`, or the simulated trap: see below); HIP `__builtin_trap()` |
| `tile.gpu_check_failed` | (emitted by `gpu_checked_value`) | store 1 to `@rocke_gpu_check_failed_<kernel>`, then `ret void`; HIP `__hip_atomic_store` + `return;`. The launch completes, `rocke.runtime` raises `GpuCheckFailed`, and every output of the launch is undefined |

`trap()` ends the wave at a trap instruction. Which instruction depends on the processor the code
object is compiled for. The docs state it here only; the docstrings and comments point here.

| Compiled for | The AMDGPU backend emits |
|---|---|
| gfx1100, gfx1101, gfx1102, gfx1103, gfx11-generic | its simulated trap: `s_trap 2`, then `s_sendmsg`, then an `s_sethalt` loop |
| any other processor, gfx1150-gfx1154 included | `s_trap 2` |

A target whose compiler does not accept its name is compiled for the generic target its catalog row
points at (`compiler_target_for_llvm_major`), and gets that target's trap: gfx1154, compiled as
gfx11-generic by compilers before LLVM 24, gets the simulated trap. `tests/core/test_trap.py` holds
the same table as its expectation and checks it on every catalogued gfx11 target.

What the host sees when a trap is taken depends on the platform:

- Linux (ROCr): the queue reports an error and the runtime aborts the process.
- Windows: on gfx110X GPUs the HIP runtime has been seen not to turn the trap into an error; the
  launch never returned.
- Under a debugger: the wave stops at the trap.

None of these lets the process carry on, so a trap is a last resort: put it under an `scf_if` on a
condition the host already checks before the launch, to stop a launch that slipped past that check
before it reaches a memory access. A check the process must survive belongs on the host, before the
launch.

A loop cannot carry a buffer descriptor (`buffer_rsrc`), as an iter_arg's initial value or as a yielded value: it is `<4 x i32>` in the IR but an opaque `ptr addrspace(8)` (HIP `rsrc_t`) once lowered, which neither a phi nor a variable can hold. The builder, the verifier and every lowering refuse it; build the descriptor before the loop and advance the offset, or rebuild it inside the loop body.

`static_for` and `unroll` are Python-time helpers that emit no `scf.for`; they execute their body in Python and emit straight-line IR.

An `scf_if_else` with `result_types` yields one value per type: each branch ends in `scf_yield` of
values of those types, and the builder's `results` are the values of the branch that ran. LLVM gets a
`phi` per result in the join block; HIP declares the results before the `if` and assigns them at the
end of both branches. A branch that does not end that way is refused by the builder, the verifier
and both lowerings. A buffer descriptor (`buffer_rsrc`) cannot be a result, for the same reason a
loop cannot carry one (above): yield what it is built from and build it after the if/else. Nor can
an LDS allocation (`smem_alloc`): allocate before the if/else and yield an offset into it. Portable
IR (the recorded recipe and the IR JSON importer) does not carry an if/else with results yet and
refuses one; lower such a kernel directly.

```python
branch = b.scf_if_else(ok, result_types=[VectorType(F16, 8)])
with branch as (then_ctx, else_ctx):
    with then_ctx:
        b.scf_yield(b.buffer_load_vN(rsrc, voffset, soffset, F16, 8))
    with else_ctx:
        b.scf_yield(b.zero_vec_f16(8))
(value,) = branch.results
```

## Compile-Time Helpers

These are not IR ops; they affect IR construction:

```text
static_for(start, stop, step, body=lambda i: ...)
unroll(stop) -> range
unroll(start, stop, step) -> range
static_if(host_bool, body)             # raises TypeError on SSA Value
```

## See Also

- `ir_lowering/ir_model.md` for the IRBuilder API by category.
- `ir_lowering/lowering_pipeline.md` for the op-to-LLVM mapping.
- `ir_lowering/backend_details.md` for purity classification and waitcnt encoding.

`IRBuilder.cvt_f32_to_tf32` / `rocke_b_cvt_f32_to_tf32` is a scalar F32-to-TF32
RNE recipe built from existing integer arithmetic, select, and bitcast ops; it
introduces no serialized opcode. It quiets NaNs and retains their upper payload
bits. `bitcast` to logical TF32 only reinterprets the payload. Logical TF32 uses
I32 storage; general arithmetic requires explicitly reinterpreting it as F32.
See [TF32 numerics](../../python/rocke/examples/gfx942/tf32_numerics/README.md).

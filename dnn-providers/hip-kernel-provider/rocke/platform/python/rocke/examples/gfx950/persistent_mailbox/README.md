# Persistent-kernel mailboxes (gfx950)

Microkernels and a harness for the cross-CTA hand-off a resident-grid
("persistent") kernel is built from: tagged `(payload, tag)` mailbox pairs
(`rocke.helpers.mailbox`) and the device ops around them
(`rocke.helpers.device_ops`).

| file | what |
|---|---|
| `kernels.py` | day-1 device-op gate, ping-pong, 160-producer all-to-all, launch-gap stamp kernel, tag-wrap kernel, bulk hand-off (rows + bf16 pairs + scoped atomics + payload_wait), device-op probe |
| `harness.py` | torch-free runner (`rocke.runtime.hip_module`), hipGraph capture for the launch gap, numerics checks, `env_problem()` |
| `bench_mailbox_rtt.py` | CLI: `gate`, `pingpong`, `a2a`, `gap`, `tagwrap`, `bulk`, `devops`, `all` |
| `mailbox_rtt_case_study.md` | the measured scope, method and relative results |

```bash
python -m rocke.examples.gfx950.persistent_mailbox.bench_mailbox_rtt all --out <dir>
```

Raw device-tick results go to `<dir>/<cmd>.json`; the console prints only
ratios and pass/fail.

## Environment

The HIP runtime the harness loads must match the COMGR rocKE compiles with.
In the q38 venv job mode (`ROCKE_COMGR_LIB` = torch's COMGR) also set

```bash
export ROCKE_HIP_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))')
```

Without it, `hipMemset` fails with out-of-memory and code objects fail with
COMGR errors. The GPU tests skip with this hint when `harness.env_problem()`
detects the mismatch. The `ir` cells also need the rocKE `library/` root on
`sys.path` (the q38 jobs set it).

## Two implementations

`MailboxKnobs.impl`:

* `asm` (default, in `rocke.helpers.mailbox`): inline asm only
  (`global_store_dwordx{2,4} ... sc1` and a self-contained asm poll loop with
  `${:uid}` labels). Both engines lower it, so a kernel built on it keeps
  cpp parity (`backend="both"` is tested).
* `ir` (in the rocKE library, `kernels.gfx950.qwen38_mono.mailbox_ir`;
  `kernels.load_ir_impl()` registers it): `IRBuilder.buffer_store_cp` and the
  `tile.tagged_poll` leaf op, which the compiler schedules. These opcodes are
  in `rocke.core.ir.PYTHON_ONLY_OPCODES`: compile with `backend="python"`
  (`harness.compile_for` picks it automatically) and exclude the build from
  cpp parity (the family README has the exemption note). It is not in
  `rocke.helpers` because `platform/AGENTS.md` forbids helpers without a C++
  port path.

The gate kernel's `ir` form (`IRBuilder.int_to_ptr`) is also Python-only and
excluded from parity.

Knobs (all default-off, named in the kernel name only when they deviate):
`impl`, `scope` (`agent` = sc1, `system` = sc0 sc1), `pause` (`s_nop`,
`s_sleep1`, `none`), `poll_max`, `fence_impl` (`ir` relies on backend
behaviour, see `device_ops.release_l2`), `put_nt`, `numa_copies`. The
all-to-all also has `barrier` (`bare` default, `sync` = `b.sync()`, the
FlyDSL reference's `gpu.barrier()` analogue).

## Protocol rules a stage author follows

* Every hand-off is one 8- or 16-byte store of `(payload, tag)` pairs; never
  split a pair across stores.
* Poll with sc1 loads only (`poll`, `get`, `pre_poll`); a plain load can spin
  forever on a stale vL1D line.
* Issue independent loads (weights) before the poll.
* No `b.sync()` around mailboxes: it drains vmcnt. `pre_poll`,
  `release_bulk` and `acquire_bulk` use `s_barrier_bare(fence="compiler")`.
* Bulk data: plain stores, then `release_bulk` (one wave fences); the consumer
  calls `acquire_bulk` before plain loads.
* Layer ids must pass `mailbox.check_layer` (layer < 63): layer 63 at
  step `2**26 - 1` produces tag 0, which zeroed memory matches.
  `launch_tag` checks a static layer itself; for a runtime layer argument
  the host launcher must call `check_layer` (`harness.run_tag_wrap` does).
* Wide builds wait with `payload_wait(rows)`: a bare barrier for at most 4
  request rows, the `pre_poll` throttle above that.

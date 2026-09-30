# gfx1250 MXF4 subtile bench (rocprofv3 ATT)

Self-contained tree for the 256x256x256 macrotile / 4096x4096x8192 problem.

```
configs/tensile_mxf4_att.yaml    Tensile ATT config (256x256, 4096^2x8192, no validate)
configs/tensile_mxf4_smoke.yaml  symlink to the common-test yaml (smoke + bench sizes)
configs/rocprof_att.yaml         rocprofv3 Advanced Thread Trace
configs/rocprof_pmc.yaml         kernel-trace + LDS PMC
configs/regress_subtile_mxf4.yaml  correctness sweep beyond the common-test shapes
logs/<stamp>/                    Tensile working dir + rocprof dumps
run.sh                           tensile | att | pmc
KNOWN_ISSUES.md                  open correctness failures
```

## Setup

Build the client once:

```bash
cd ../../   # tensilelite root
invoke build-client --gpu-targets gfx1250
```

The system `python3` has no `joblib`/`msgpack`, so Tensile needs a venv. `run.sh` picks up
`tensilelite/.venv` automatically if it exists:

```bash
cd ../../
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

`rocisa` is *not* pip-installed; `run.sh` puts `build_tmp/tensilelite/rocisa` on `PYTHONPATH`,
which is where `invoke build-client` leaves the compiled `_rocisa*.so`.

## Run

```bash
cd bench/gfx1250_mxf4
# Pins physical GPU 3 via HIP_VISIBLE_DEVICES (override with GPU_ID=N).
./run.sh tensile          # compile + one timed launch, ISA under tensile/ (KeepBuildTmp)
./run.sh att              # ATT on the latest ClientParameters.ini
./run.sh pmc              # TX_VMW LDS bank / address / segment-conflict counters
./run.sh att --tensile    # wrap the whole Tensile process instead of the client
```

Each invocation creates `logs/YYYYMMDD-HHMMSS/`. ATT writes `rocprof/`; Tensile writes `tensile/`.

Only one GPU filter is set. `ROCR_VISIBLE_DEVICES` already renumbers the surviving card to 0,
so setting it *and* `HIP_VISIBLE_DEVICES=3` filters the filtered list and the client dies with
`hipErrorNoDevice`.

## Enabling the TX_VMW LDS counters

The `TX_PERF_SEL_VMW_*` counters read 0 until a per-XCC register is set on the card being
profiled. umr's `-i` is the DRM instance, not the HIP device index; map them with
`sudo cat /sys/kernel/debug/dri/<N>/name` against `rocm-smi --showbus`. On this box GPU 3
(PCI `0004:01:00.0`) is instance 25 (GPU 0 is 1, GPU 1 is 9, GPU 2 is 17):

```bash
for r in 0x3b120 0x7b120 0xbb120 0xfb120 0x13b120 0x17b120 0x1bb120 0x1fb120; do
  sudo env LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib umr -i 25 -go 0 -vmp -w $r 0x400
done
```

Check liveness with the positive control (`control/lds_bank_conflict` under
`configs/rocprof_lds_control.yaml`): `lds_conflict` must show a nonzero
`TX_PERF_SEL_VMW_LDS_BANK_CONFLICT`.

## gfx1250 profiling caveats

`./run.sh att` completes and the decoder emits full per-code-object disassembly
(`att_gfx1250_code_object_id_*.out`), but the instruction table
(`stats_ui_output_*.csv`) comes back header-only — no hitcount/latency/stall rows.
That held across `att_target_cu` 0/2/4, `att_shader_engine_mask` 0x1/0xF/0xFF, and
`att_simd_select` 0x0/0xF. `--att-perfcounters` is gfx9-only.

`SQ_LDS_BANK_CONFLICT` / `SQC_LDS_BANK_CONFLICT` are gfx9/gfx12 only. On gfx1250 the LDS
conflict events live on the TCP/VMW perf selects (`TX_PERF_SEL_VMW_*` in the Arcadia headers).
`configs/rocprof_pmc.yaml` collects:

| Counter | Event | What it counts |
| --- | --- | --- |
| `TX_VMW_LDS_BANK_CONFLICT` | 0x84 | Cycles LDS is stalled by a bank conflict |
| `TX_VMW_LDS_ADDR_CONFLICT` | 0x87 | Cycles LDS is stalled by an address conflict |
| `TX_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES` | 0xa4 | Cycles LDS is stalled because both read ports hit the same segment |

`SQ_INST_CYCLES_LDS / SQ_INSTS_LDS` is still collected. On the current subtile kernel both
read **0** — the TDM path may not increment `SQ_INSTS_LDS` — so use the `TX_VMW_LDS_*` counters
and `SQ_BUSY_CYCLES` / kernel time. Confirm the names on the box with:

```bash
rocprofv3-avail info --pmc | rg -i 'vmw_lds|segment_conflict'
```

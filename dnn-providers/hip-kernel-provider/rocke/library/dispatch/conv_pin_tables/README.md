# Conv pin tables

A pin table names, for each convolution problem it covers and for each selection mode, the AOT kernel instance and
launch parameters that dispatch should use. It is read by `dispatch/conv_pins.py`. A table is an opt-in dispatch
candidate: requests with `algorithm="auto"` never select it.

| file | arch | problems | modes |
|---|---|---|---|
| `gfx942_unet3d_bf16.json` | gfx942 | 3-D UNet convolutions: bf16, NDHWC, 3x3x3 filter, stride 1; 10 shapes x fwd / dgrad / wgrad, each in the halo form (no padding, cases `<dir>_s<N>`) and the adapter form (padding 1, cases `<dir>_s<N>_pad1`) | `deterministic`, `fastest` |

## Format

```json
{"arch": "gfx942", "llvm_flavor": "llvm22", "comment": "...",
 "modes": {"<mode>": {"deterministic": true,
                      "cases": {"<case>": {"cmd": "MIOpenDriver convbfp16 ... -F <1|2|4> ...",
                                           "calls": 2,
                                           "kernel": "<identity label> [@split_k=<d>[/full<t>]]",
                                           "sessions": 12,
                                           "deterministic": true}}}}}
```

- A case is keyed by its problem: direction, dtype, and every dimension, stride, pad and dilation, parsed from `cmd`.
  A request matches a case only when all of these are equal.
- `kernel` is the lossless identity label of an AOT binary (`KernelIdentity.from_label`), optionally followed by the
  launch it was selected with: the split-K degree and, for a tail split, the number of full tiles.
- `deterministic` on a case records the `benchmarks/common/det_classify.py` verdict (`f32`) for that launch. A mode
  marked `deterministic` holds only such launches; `conv_pin_table.py` refuses to write one that does not.
- Tables hold the selection only. The measurements behind each pick stay with the run that made it.

## Using a table

```python
from dispatch.conv_pins import dispatch_conv_pinned, load_conv_pins, request_from_miopen_cmd

pins = load_conv_pins("dispatch/conv_pin_tables/gfx942_unet3d_bf16.json", "deterministic")
res = dispatch_conv_pinned(request_from_miopen_cmd(cmd, "gfx942"), pins)  # ValueError: not pinned -> fall back
res.spec.identity        # AOT cache key of the binary
res.spec.launch_params() # {"split_k": ..., "tail_full": ...}
```

`register_conv_pin_table(pins)` adds the fwd and dgrad candidates to the grouped-convolution registries, which makes
`dispatch_conv_grouped(replace(req, algorithm="pinned"))` return the pinned instance. The step replay
(`benchmarks/common/conv_step_replay.py --pins <table>#<mode>`, or `<table>#<mode>@pad1` for the `_pad1` cases) resolves
every case through `dispatch_conv_pinned`.

A pinned binary admits only problems its identity serves: halo-form kernels (`uk_nopad`, `tapk_xhalo`) do not admit
padded problems, which is why the padded form has cases of its own. A request the table does not pin misses it and
falls back to the default selection.

## Regenerating a table

1. Collect candidates for each case from the finalists of `benchmark_conv_compare.py --fast-rank` runs over the AOT
   cache. Add `--deterministic f32` for the deterministic mode.
2. Judge them in several interleaved sessions, ideally spread over GPUs: `conv_step_replay.py --mode ab` with one
   `--pins-variant` per candidate. For each case, keep the candidate with the lowest mean time relative to the
   reference pick, and switch only when its time is consistently lower.
3. Validate the result against the reference picks, in staged per-call timing and in the sustained step
   (`--mode ab --mode step`). Then run a pinned `--verify` run of every launch.
4. Write the table: `python benchmarks/common/conv_pin_table.py --out <table> --arch gfx942
   --mode deterministic=<pins> --mode fastest=<pins>`.

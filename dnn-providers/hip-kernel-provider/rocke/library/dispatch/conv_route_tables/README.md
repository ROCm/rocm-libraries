# Conv route tables

A route table names, for each convolution problem it covers and for each selection mode, the *route* that serves a
call: one pinned kernel, a pad copy plus a pinned halo-form kernel, or the 3-D Winograd drop-in. It is read by
`dispatch/conv_routes.py`. Route tables are opt-in: nothing reads one unless a caller passes it, and default dispatch
(`algorithm="auto"`, the pin tables) is unchanged.

| file | arch | problems | modes |
|---|---|---|---|
| `gfx942_unet3d_bf16.json` | gfx942 | 3-D UNet convolutions: bf16, NDHWC, 3x3x3 filter, stride 1; 10 shapes x fwd / dgrad / wgrad in the padded forms: padding (1,1,1) (cases `<dir>_s<N>_pad1`) and padding (0,1,1) with the depth halo in the input (cases `<dir>_s<N>_pad011`) | `deterministic` |

## Routes

| route | directions | launches | workspaces |
|---|---|---|---|
| `direct` | fwd, dgrad, wgrad | the pinned kernel on the padded problem | its own launch scratch only |
| `padcopy_halo` | fwd | `pad_copy` (X -> interior of Xp), then the pinned halo-form kernel on Xp | `Xp` [D+2pd][H+2ph][W+2pw][C] bf16, halo zeroed once |
| `winograd` | fwd, dgrad | aux (relayout to c8 with the zero halo + weight transform), yx main kernel, split-K reduction when `ksplit` > 1 | `X8` (zeroed once), `U`, `E` (prescale map), `P` (split-K partials) |

The halo-form kernel of a `padcopy_halo` route runs the *halo problem*: the padded problem with its padding folded into
the input extents and padding 0 (`conv_routes.halo_request`). On the zero-haloed copy it computes the padded
convolution with the same products and reduction order. A Winograd route keeps the drop-in's error class (f16
operands with a per-block prescale); every route is bitwise deterministic run to run.

## Format

```json
{"arch": "gfx942", "llvm_flavor": "llvm22", "comment": "...",
 "modes": {"<mode>": {"deterministic": true,
                      "cases": {"<case>": {"cmd": "MIOpenDriver convbfp16 ... -F <1|2|4> ...",
                                           "calls": 2,
                                           "route": "direct | padcopy_halo | winograd",
                                           "kernel": "<identity label> [@split_k=<d>[/full<t>]]",
                                           "winograd": {"spec": {"<WinogradYxSpec field>": "...", "...": "..."},
                                                        "wtv": 4}}}}}}
```

- A case is keyed by its problem exactly as a pin is (`conv_pins.request_key`): direction, dtype, and every dimension,
  stride, pad and dilation, parsed from `cmd` -- the padded problem the call is. One problem has one route per mode.
- `kernel` (`direct`, `padcopy_halo`) is a pin label; for `padcopy_halo` it is the halo-form kernel. A row without
  `kernel` takes the kernel of a pin table passed with the routes (`pins=`): the padded problem's pin (`direct`) or the
  halo problem's pin (`padcopy_halo`).
- `winograd.spec` holds the `kernels.common.conv_winograd3d.WinogradYxSpec` fields except the extents, which follow
  from the problem; `wtv` is the weight-transform variant. The prescale halo of the spec (`bps_halo`, `bps_halo_z`)
  must equal the relayout halo of the problem (fwd: the padding; dgrad: 2 - the padding).
- `calls` is the number of calls per training step (replay weight).

## Using a table

```python
from dispatch.conv_routes import dispatch_conv_route, load_conv_routes, request_for_route
from dispatch.conv_pins import request_from_miopen_cmd

routes = load_conv_routes("dispatch/conv_route_tables/gfx942_unet3d_bf16.json", "deterministic")
plan = dispatch_conv_route(request_from_miopen_cmd(cmd, "gfx942"), routes)  # ValueError: not routed -> fall back
plan.route            # "direct" | "padcopy_halo" | "winograd"
plan.steps            # RouteStep launches in order: kind "kernel" (build(arch) -> KernelDef, signature, grid,
                      # block, args -> buffers) or "pinned" (plan.kernel on plan.kernel_request)
plan.workspaces       # RouteWorkspace(name, nbytes, zero_once)
plan.kernel           # DispatchResult of the pinned kernel (direct / padcopy_halo), via dispatch_conv_pinned
plan.winograd         # Winograd3dPlan (winograd): Winograd3dDropin(plan.problem, plan.direction, plan.winograd)
```

`dispatch_conv_routed(req, routes, pins)` answers from both opt-in tables, routes first: the route when the route
table covers the problem, else a `direct` plan of the pin table's kernel (`ValueError` when neither does, or when the
two tables are of different modes).

A route serves NDHWC requests of the table's arch and, when the table names one, its LLVM flavor, with `algorithm`
`"auto"` or `"pinned"`; anything else raises `ValueError`. When loading, a `padcopy_halo` or `winograd` row needs a
bf16 command (the pad copy and the drop-in are bf16), and a named `kernel` must be of its command's direction and dtype.
The mode's `deterministic` flag is kept on the loaded routes (`ConvRoutes.deterministic`).

`benchmarks/common/conv_route_exec.RouteBinding` binds a plan to device buffers (torch-free; the pinned step is bound
as `--run-from-cache` binds a cached binary). `benchmarks/common/conv_route_check.py` verifies, determinism-checks and
times every routed case through dispatch; `benchmarks/common/conv_step_replay.py --routes <table>#<mode>@pad1`
replays a training step through the routes (`--routes-variant NAME:<table>#<mode>@<form>` judges them against
`--pins` in one session; `--form-variant NAME:<pin table>#<mode>` adds the halo form of the same step, on its own
buffers, as a reference).

## Regenerating a table

1. Judge the routes of each padded case in interleaved same-session rounds: the pinned direct kernel, the pinned
   halo-form kernel on a pad copy, and the Winograd drop-in candidates (Winograd benchmark `--adapter`, with `--verify` and
   repeated-run determinism).
2. Keep each case's fastest route; write the rows above (the halo-form kernel label, or the Winograd spec and `wtv`).
   The `kernel` of a `direct` / `padcopy_halo` row is the deterministic pick of the shipped pin table
   (`dispatch/conv_pin_tables`): the padded problem's pin, or the halo problem's pin. Regenerate the labels after a
   pin table refresh (drop `kernel`, resolve with `pins=`, write the plan's `kernel_label` back);
   `tests/test_conv_routes.py` checks that the two agree.
3. Validate through dispatch: `conv_route_check.py --verify --det-runs N --rounds R` on every case, the output
   digests compared across two devices (`--against`), and the step replay.

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Production routes of padded convolutions (opt-in route tables).

A *route table* says, per convolution problem, HOW a call is served -- not
only which kernel runs it. A padded problem (the production form of a 3-D
UNet: padding (1,1,1), or (0,1,1) on a depth-split run whose input carries its
depth halo) can be served three ways:

``direct``
    One pinned implicit-GEMM launch on the padded problem itself (what
    :mod:`dispatch.conv_pins` serves).
``padcopy_halo``
    (fwd) A rocKE pad copy (:func:`kernels.common.conv_winograd3d.build_pad_copy`)
    of the input into a zero-haloed workspace, then a pinned *halo-form* kernel
    (padding 0, input = output + 2) on that workspace. The halo-form kernels
    refuse padded problems; on the zero-haloed copy they compute the padded
    convolution exactly (same products, same reduction order).
``winograd``
    The 3-D yx Winograd drop-in
    (:class:`kernels.common.conv_winograd3d_dropin.Winograd3dDropin`, fwd and
    dgrad) with a stored kernel spec: relayout + weight transform, main kernel,
    and the ordered split-K reduction when the spec splits K.

Every route is deterministic when its kernels are (the pad copy and the
Winograd drop-in are: plain stores, ordered reductions).

Table format (one file, several selection modes, like a pin table)::

    {"arch": "gfx942", "llvm_flavor": "llvm22", "comment": "...",
     "modes": {"deterministic": {"deterministic": true, "cases": {
        "fwd_s1_pad1":   {"cmd": "MIOpenDriver ... -F 1 ...", "calls": 2,
                          "route": "padcopy_halo",
                          "kernel": "<halo-form identity label> [@split_k=..]"},
        "fwd_s3_pad1":   {"cmd": "...", "route": "winograd",
                          "winograd": {"spec": {...WinogradYxSpec fields...}, "wtv": 4}},
        "dgrad_s1_pad1": {"cmd": "...", "route": "direct",
                          "kernel": "<identity label> [@split_k=..]"}}}}}

* ``cmd`` is the MIOpenDriver command of the problem the route serves (the
  padded problem); cases are keyed by it exactly as pins are
  (:func:`dispatch.conv_pins.request_key`): direction, dtype and every
  dimension / stride / pad / dilation.
* ``kernel`` (direct, padcopy_halo) is a pin label: the AOT identity plus the
  launch it was selected with. ``padcopy_halo`` names the halo-form kernel, which runs
  the *halo problem* (:func:`halo_request`). Leaving ``kernel`` out defers to
  a pin table passed to :func:`dispatch_conv_route` (``pins=``), looked up by
  the padded problem (direct) or the halo problem (padcopy_halo).
* ``winograd.spec`` holds the :class:`~kernels.common.conv_winograd3d.WinogradYxSpec`
  fields except the extents (derived from the problem), ``winograd.wtv`` the
  weight-transform variant.

Nothing reads a route table unless a caller passes one: default dispatch is
unchanged. :func:`dispatch_conv_route` raises ``ValueError`` for a request the
table does not route; the caller then falls back to its default selection.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .conv_pins import (
    ConvPin,
    ConvPins,
    _SPLIT_RE,
    dispatch_conv_pinned,
    parse_miopen_cmd,
    request_key,
    table_modes,
)

ROUTE_DIRECT = "direct"
ROUTE_PADCOPY_HALO = "padcopy_halo"
ROUTE_WINOGRAD = "winograd"
ROUTES = (ROUTE_DIRECT, ROUTE_PADCOPY_HALO, ROUTE_WINOGRAD)

#: Unroll of the route's pad copy (16-byte units per thread).
PAD_COPY_UNROLL = 4

#: Spec fields the table never carries (derived from the problem) or the
#: Winograd benchmark's selector field.
_WINO_DERIVED = ("Di", "Hi", "Wi", "C", "K", "name", "variant")


@dataclass(frozen=True)
class ConvRoute:
    """One routed case of a table."""

    case: str
    route: str
    cmd: str
    kernel: str | None = None
    winograd_spec: tuple = ()  # sorted (field, value) items
    wtv: int = 0
    calls: int = 1
    ms: float | None = None

    @property
    def spec_dict(self) -> dict:
        return dict(self.winograd_spec)


@dataclass
class ConvRoutes:
    """One mode of a route table, keyed by problem."""

    source: str = ""
    routes: dict = field(default_factory=dict)  # request_key -> ConvRoute
    arch: str = ""
    mode: str = ""
    llvm_flavor: str = ""
    _pinsets: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.routes)

    def lookup(self, req) -> ConvRoute | None:
        if self.arch and getattr(req, "arch", self.arch) != self.arch:
            return None
        return self.routes.get(request_key(req))

    def by_case(self) -> dict:
        return {r.case: r for r in self.routes.values()}


def _jsonable(v):
    return list(v) if isinstance(v, tuple) else v


def route_from_row(case: str, row: Mapping, source: str = "") -> ConvRoute:
    """A :class:`ConvRoute` from one decoded table row (validated)."""
    route = str(row.get("route", ROUTE_DIRECT)).strip().lower()
    if route not in ROUTES:
        raise ValueError(f"{source}: case {case}: route {route!r} (one of {', '.join(ROUTES)})")
    cmd = str(row["cmd"])
    direction, _dtype, key = parse_miopen_cmd(cmd)
    kernel = row.get("kernel")
    kernel = str(kernel).strip() if kernel else None
    spec, wtv = (), 0
    if route == ROUTE_WINOGRAD:
        w = row.get("winograd")
        if not isinstance(w, Mapping) or not isinstance(w.get("spec"), Mapping):
            raise ValueError(f"{source}: case {case}: a winograd route needs winograd.spec")
        if direction not in ("fwd", "dgrad"):
            raise ValueError(f"{source}: case {case}: winograd serves fwd / dgrad, not {direction}")
        bad = sorted(set(w["spec"]) & set(_WINO_DERIVED) - {"variant"})
        if bad:
            raise ValueError(f"{source}: case {case}: winograd.spec carries derived fields {bad}")
        if w["spec"].get("variant", "yx") != "yx":
            raise ValueError(f"{source}: case {case}: winograd.spec variant must be yx")
        spec = tuple(sorted((k, _jsonable(v)) for k, v in w["spec"].items() if k != "variant"))
        wtv = int(w.get("wtv", 1))
        if kernel:
            raise ValueError(f"{source}: case {case}: a winograd route names no pinned kernel")
    elif route == ROUTE_PADCOPY_HALO:
        if direction != "fwd":
            raise ValueError(f"{source}: case {case}: padcopy_halo serves fwd, not {direction}")
        names = dict(zip(("N", "C", "K", "Hi", "Wi", "Y", "X", "G", "stride_h", "stride_w", "pad_h",
                          "pad_w", "dilation_h", "dilation_w", "Di", "Z", "stride_d", "pad_d",
                          "dilation_d"), key))
        if not any(names[p] for p in ("pad_d", "pad_h", "pad_w")):
            raise ValueError(f"{source}: case {case}: padcopy_halo needs a padded problem")
    return ConvRoute(
        case=case,
        route=route,
        cmd=cmd,
        kernel=kernel,
        winograd_spec=spec,
        wtv=wtv,
        calls=int(row.get("calls", 1)),
        ms=float(row["ms"]) if row.get("ms") is not None else None,
    )


def routes_from_mapping(data: Mapping, source: str = "") -> ConvRoutes:
    """:class:`ConvRoutes` from a decoded mode (``{"cases": {...}}``)."""
    out = ConvRoutes(source=source)
    for case, row in (data.get("cases") or {}).items():
        r = route_from_row(case, row, source)
        key = parse_miopen_cmd(r.cmd)
        if key in out.routes:
            raise ValueError(f"{source}: cases {out.routes[key].case} and {case} route one problem")
        out.routes[key] = r
    return out


def load_conv_routes(path, mode: str | None = None) -> ConvRoutes:
    """Read one ``mode`` of a route table (the only mode when ``mode`` is None)."""
    p = Path(path)
    data = json.loads(p.read_text())
    modes = table_modes(data)
    if not modes:
        raise ValueError(f"{p}: not a route table (no modes)")
    if mode is None:
        if len(modes) != 1:
            raise ValueError(f"{p}: route table with modes {', '.join(modes)}: name one")
        mode = modes[0]
    if mode not in modes:
        raise ValueError(f"{p}: no mode {mode!r} (modes: {', '.join(modes)})")
    out = routes_from_mapping(data["modes"][mode], source=f"{p}#{mode}")
    out.arch = str(data.get("arch", "") or "")
    out.mode = mode
    out.llvm_flavor = str(data.get("llvm_flavor", "") or "")
    return out


# ---------------------------------------------------------------------------
# Route plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PadCopySpec:
    """The pad copy of a ``padcopy_halo`` route: X [D][H][W][C] bf16 -> the
    interior of Xp [D+2hz][H+2hy][W+2hx][C] (whose halo stays zero)."""

    D: int
    H: int
    W: int
    C: int
    halo: tuple  # (hz, hy, hx)
    unroll: int = PAD_COPY_UNROLL

    @property
    def out_shape(self) -> tuple:
        hz, hy, hx = self.halo
        return (self.D + 2 * hz, self.H + 2 * hy, self.W + 2 * hx, self.C)

    @property
    def out_bytes(self) -> int:
        n = 2
        for d in self.out_shape:
            n *= d
        return n

    def build(self, arch: str = "gfx942"):
        """The pad-copy ``KernelDef`` (``arch``-independent IR)."""
        from kernels.common import conv_winograd3d as cw

        return cw.build_pad_copy(self.D, self.H, self.W, self.C, halo=tuple(self.halo),
                                 unroll=self.unroll)

    def grid(self) -> tuple:
        from kernels.common import conv_winograd3d as cw

        return cw.pad_copy_grid(self.D, self.H, self.W, self.C, unroll=self.unroll)

    block = (256, 1, 1)

    @staticmethod
    def signature() -> tuple:
        from rocke.helpers.spec import SignatureBuilder

        return tuple(SignatureBuilder().ptr("X", "bf16").ptr("Xp", "bf16").build())


@dataclass(frozen=True)
class RouteWorkspace:
    """A device buffer a route needs beside the call's own tensors.

    ``zero_once``: allocate it zeroed; the route never writes the part that
    must stay zero (a pad copy's halo, the relayout's halo), so it is zeroed
    once per allocation, not per call. Launch scratch of a pinned kernel (its
    split-K workspace, zeroed outputs) is sized by that kernel's launcher and
    is not listed here."""

    name: str
    nbytes: int
    zero_once: bool = False


@dataclass(frozen=True)
class RouteStep:
    """One launch of a route, in order.

    ``kind="kernel"``: a rocKE kernel built for the problem -- ``build(arch)``
    returns its ``KernelDef`` (compile it with ``rocke.helpers.compile_kernel``),
    launched with ``signature`` / ``grid`` / ``block``; ``args`` maps each
    kernel parameter to a buffer: the call's tensors ``x`` / ``w`` / ``y``
    (fwd X, W, Y; dgrad dY, W, dX) or a workspace by name.

    ``kind="pinned"``: the plan's pinned implicit-GEMM kernel, bound as a
    pinned kernel is (its own launch scratch / zeroing) on
    ``plan.kernel_request``; ``args`` maps its A / B / D operands to buffers.
    """

    name: str
    kind: str  # "kernel" | "pinned"
    args: tuple  # ((param, buffer), ...)
    build: object = None  # callable(arch) -> KernelDef ("kernel")
    signature: tuple = ()
    grid: tuple = ()
    block: tuple = (256, 1, 1)

    @property
    def arg_map(self) -> dict:
        return dict(self.args)


@dataclass(frozen=True)
class ConvRoutePlan:
    """What :func:`dispatch_conv_route` hands a runtime for one call.

    * ``direct``: ``kernel`` is the pinned ``DispatchResult`` on ``request``.
    * ``padcopy_halo``: ``pad_copy`` X -> workspace ``Xp``, then ``kernel``
      (pinned halo-form ``DispatchResult``) on ``kernel_request`` with Xp as
      its input.
    * ``winograd``: ``winograd`` is the drop-in plan
      (``Winograd3dDropin(problem, direction, winograd)``).
    """

    case: str
    route: str
    request: object
    problem: object  # ConvProblem of request
    kernel: object | None = None  # DispatchResult (direct / padcopy_halo)
    kernel_request: object | None = None
    pad_copy: PadCopySpec | None = None
    winograd: object | None = None  # Winograd3dPlan
    workspaces: tuple = ()
    steps: tuple = ()  # RouteStep, in launch order
    calls: int = 1

    @property
    def kernel_problem(self):
        """ConvProblem the pinned kernel runs (the halo problem for padcopy_halo)."""
        from .grouped_convolution import _conv_problem

        return _conv_problem(self.kernel_request) if self.kernel_request is not None else None

    @property
    def launches(self) -> tuple:
        return tuple(st.name for st in self.steps)

    @property
    def direction(self) -> str:
        return str(self.request.direction).lower()

    @property
    def workspace_bytes(self) -> int:
        return sum(w.nbytes for w in self.workspaces)

    @property
    def kernel_label(self) -> str:
        """Pin label (identity [@split_k=..]) of the pinned kernel, '' for winograd."""
        if self.kernel is None:
            return ""
        lp = self.kernel.spec.launch_params()
        s = self.kernel.spec.name
        if "split_k" in lp:
            s += f" @split_k={lp['split_k']}" + (f"/full{lp['tail_full']}" if "tail_full" in lp else "")
        return s

    def describe(self) -> str:
        return f"{self.case}: {self.route} [" + " -> ".join(self.launches) + "]"


def halo_request(req):
    """The halo-form request of a padded ``req``: padding folded into the
    input extents (D + 2 pd, H + 2 ph, W + 2 pw), padding 0 -- the problem a
    halo-form kernel runs on the pad copy."""
    pd = int(req.pad_d or 0) if req.Di is not None else 0
    ph, pw = int(req.pad_h), int(req.pad_w)
    kw = dict(Hi=int(req.Hi) + 2 * ph, Wi=int(req.Wi) + 2 * pw, pad_h=0, pad_w=0)
    if req.Di is not None:
        kw.update(Di=int(req.Di) + 2 * pd, pad_d=0)
    return dataclasses.replace(req, **kw)


def _pin_dispatch(routes: ConvRoutes, req, label: str, case: str):
    """Dispatch ``req`` to the pinned ``label`` through :mod:`dispatch.conv_pins`
    (so the route's kernel passes the same admission as a pinned one). The
    one-case pins set is kept on ``routes`` (one dispatch registry per pin)."""
    key = (request_key(req), label)
    pins = routes._pinsets.get(key)
    if pins is None:
        m = _SPLIT_RE.search(label)
        pins = ConvPins(source=f"{routes.source}:{case}", arch=routes.arch or str(req.arch),
                        mode=routes.mode or "routes")
        pins.pins[key[0]] = ConvPin(
            case=case,
            instance=label.split()[0],
            label=label,
            split_k=int(m.group(1)) if m else None,
            tail_full=int(m.group(2)) if m and m.group(2) is not None else None,
        )
        routes._pinsets[key] = pins
    return dispatch_conv_pinned(req, pins)


def _pins_label(pins: ConvPins | None, req) -> str | None:
    if pins is None:
        return None
    pin = pins.lookup(req)
    return pin.label if pin is not None else None


def winograd_route_plan(route: ConvRoute, problem, direction: str):
    """The :class:`Winograd3dPlan` of a ``winograd`` route (validated)."""
    from kernels.common import conv_winograd3d as cw
    from kernels.common import conv_winograd3d_select as sel

    why = sel.winograd3d_ineligible_reason(problem, direction)
    if why:
        raise ValueError(f"{route.case}: winograd route on an ineligible problem: {why}")
    spec_kw = route.spec_dict
    Di, Hi, Wi, C, K = sel._winograd_problem(problem, direction)
    spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **spec_kw)
    ok, why = cw.is_valid_yx_spec(spec)
    if not ok:
        raise ValueError(f"{route.case}: invalid winograd spec: {why}")
    hz, h = sel._relayout_halos(problem, direction)
    if spec.bps not in (0, 1) or spec.oscale or spec.yc8h:
        raise ValueError(f"{route.case}: drop-in needs bps 0 / 1, no oscale / yc8h")
    if spec.bps == 1 and (spec.bps_halo != h or cw._bps_hz(spec) != hz):
        raise ValueError(f"{route.case}: bps halo ({spec.bps_halo}, z {cw._bps_hz(spec)}) is not the "
                         f"relayout halo ({h}, z {hz})")
    return sel.Winograd3dPlan(
        "yx", tuple(sorted(spec_kw.items())), route.wtv, "dropin", direction,
        "bps" if spec.bps else "none", source="route table", reason=route.case,
    ), spec


def winograd_workspaces(problem, direction: str, spec) -> tuple:
    """Workspaces of :class:`Winograd3dDropin` for ``spec`` (same sizes it allocates)."""
    from kernels.common import conv_winograd3d as cw
    from kernels.common import conv_winograd3d_select as sel

    hz, h = sel._relayout_halos(problem, direction)
    src = (spec.Di - 2 * hz, spec.Hi - 2 * h, spec.Wi - 2 * h, spec.C)
    out = [RouteWorkspace("X8", spec.C * spec.Di * spec.Hi * spec.Wi * 2, zero_once=True),
           RouteWorkspace("U", 3 * spec.nbp * 4 * spec.C * spec.K * 2)]
    if spec.bps == 1:
        out.append(RouteWorkspace("E", cw.x_to_c8_lds_grid(*src)[0] * 4, zero_once=True))
    if spec.ksplit > 1:
        out.append(RouteWorkspace("P", spec.ksplit * spec.Do * spec.Ho * spec.Wo * spec.K * 4))
    return tuple(out)


def winograd_steps(plan, problem, direction: str) -> tuple:
    """The launches of :class:`Winograd3dDropin` for ``plan`` (the same kernels,
    signatures, grids and buffers): aux (relayout + weight transform), main,
    and the split-K reduction when the spec splits K."""
    from rocke.helpers.spec import SignatureBuilder

    from kernels.common import conv_winograd3d as cw
    from kernels.common import conv_winograd3d_select as sel

    hz, h = sel._relayout_halos(problem, direction)
    Di, Hi, Wi, C, K = sel._winograd_problem(problem, direction)
    spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **plan.spec_dict)
    src = (Di - 2 * hz, Hi - 2 * h, Wi - 2 * h, C)
    halo = h if hz == h else (hz, h, h)
    odt = "f16" if spec.op_dtype == "f16" else "bf16"
    wtv = plan.wtv or 1
    wt3 = wtv in (3, 4) and odt == "f16" and spec.stage == "lds16" and not spec.b16 and not spec.bc
    wv = wtv if wt3 else 1
    bps1 = spec.bps == 1
    ks = spec.ksplit

    def aux(arch: str = "gfx942"):
        return cw.build_yx_aux(*src, K, out_dtype=odt, fold=spec.stage in ("reg", "wlds"), rev=spec.xrev,
                               halo=halo, flip=direction == "dgrad", wtv=wv, amax=bps1, my=spec.my)

    sga = SignatureBuilder().ptr("X", "bf16").ptr("X8", "bf16").ptr("W", "bf16").ptr("U", odt)
    sgm = SignatureBuilder().ptr("X", "bf16").ptr("U", odt).ptr("Y", "f32" if ks > 1 else "bf16")
    ea = ()
    if bps1:
        sga, sgm, ea = sga.ptr("E", "i32"), sgm.ptr("E", "i32"), (("E", "E"),)
    steps = [
        RouteStep("winograd aux (relayout + weight transform)", "kernel",
                  (("X", "x"), ("X8", "X8"), ("W", "w"), ("U", "U")) + ea, build=aux,
                  signature=tuple(sga.build()), grid=cw.yx_aux_grid(*src, K, wtv=wv)),
        RouteStep(f"winograd main {spec.name}", "kernel",
                  (("X", "X8"), ("U", "U"), ("Y", "P" if ks > 1 else "y")) + ea,
                  build=lambda arch="gfx942": cw.build_winograd_yx(spec), signature=tuple(sgm.build()),
                  grid=cw.winograd_yx_grid(spec), block=(spec.block_size, 1, 1)),
    ]
    if ks > 1:
        nel = spec.Do * spec.Ho * spec.Wo * K
        steps.append(RouteStep(
            f"split-K reduce ({ks})", "kernel", (("P", "P"), ("Y", "y")),
            build=lambda arch="gfx942": cw.build_ks_reduce(nel, ks),
            signature=tuple(SignatureBuilder().ptr("P", "f32").ptr("Y", "bf16").build()),
            grid=cw.ks_reduce_grid(nel)))
    return tuple(steps)


def dispatch_conv_route(req, routes: ConvRoutes, pins: ConvPins | None = None) -> ConvRoutePlan:
    """The production route of ``req`` (a ``ConvGroupedRequest``, any direction).

    Raises ``ValueError`` when the table does not route the problem or the
    route cannot serve it (the caller falls back to its default selection).
    ``pins`` serves routes whose row names no kernel (and nothing else).
    """
    from .grouped_convolution import _conv_problem

    route = routes.lookup(req)
    if route is None:
        raise ValueError(f"{routes.source or 'routes'}: no route for this problem")
    problem = _conv_problem(req)
    if problem is None:
        raise ValueError("not a valid convolution shape")
    direction = str(req.direction).lower()
    base = dict(case=route.case, route=route.route, request=req, problem=problem, calls=route.calls)
    if route.route == ROUTE_DIRECT:
        label = route.kernel or _pins_label(pins, req)
        if not label:
            raise ValueError(f"{route.case}: direct route names no kernel and the pins have none")
        res = _pin_dispatch(routes, req, label, route.case)
        step = RouteStep(f"pinned {res.spec.name}", "pinned", (("A", "x"), ("B", "w"), ("D", "y")))
        return ConvRoutePlan(**base, kernel=res, kernel_request=req, steps=(step,))
    if route.route == ROUTE_PADCOPY_HALO:
        hreq = halo_request(req)
        label = route.kernel or _pins_label(pins, hreq)
        if not label:
            raise ValueError(f"{route.case}: padcopy_halo route names no kernel and the pins have none "
                             "for the halo problem")
        res = _pin_dispatch(routes, hreq, label, route.case)
        p = problem
        pc = PadCopySpec(D=p.Di, H=p.Hi, W=p.Wi, C=p.C, halo=(p.pD, p.pH, p.pW))
        steps = (RouteStep(f"pad_copy halo {pc.halo}", "kernel", (("X", "x"), ("Xp", "Xp")),
                           build=pc.build, signature=pc.signature(), grid=pc.grid(), block=pc.block),
                 RouteStep(f"pinned {res.spec.name} (halo form)", "pinned",
                           (("A", "Xp"), ("B", "w"), ("D", "y"))))
        return ConvRoutePlan(**base, kernel=res, kernel_request=hreq, pad_copy=pc,
                             workspaces=(RouteWorkspace("Xp", pc.out_bytes, zero_once=True),),
                             steps=steps)
    plan, spec = winograd_route_plan(route, problem, direction)
    return ConvRoutePlan(**base, winograd=plan, workspaces=winograd_workspaces(problem, direction, spec),
                         steps=winograd_steps(plan, problem, direction))


def request_for_route(route: ConvRoute, arch: str, **kw):
    """The ``ConvGroupedRequest`` of a routed case (from its command)."""
    from .conv_pins import request_from_miopen_cmd

    return request_from_miopen_cmd(route.cmd, arch, **kw)


#: Shipped route tables (``<arch>_<name>.json``), beside this module.
ROUTE_TABLE_DIR = Path(__file__).resolve().with_name("conv_route_tables")


def shipped_conv_route_tables() -> dict:
    """The route tables shipped beside this module, by file stem."""
    if not ROUTE_TABLE_DIR.is_dir():
        return {}
    return {f.stem: f for f in sorted(ROUTE_TABLE_DIR.glob("*.json"))}

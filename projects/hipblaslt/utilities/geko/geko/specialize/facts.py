# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-call facts for ``--specialize``.

Generation facts (types, sizes, transposes, leading dimensions, strides, route)
must be logged or proven; a missing one refuses the call. Narrowing facts
(beta, activation, bias, scale-alpha vector, pointer mode) that are unknown or
unproven withhold exactly the narrowing they would enable; the call is still
generated with the stock capability and the withheld fact is reported, unless the
retained capability is a selection fact the call does not meet (a kept ``UseBias``
requires bias presence; the writer then refuses the call). Values always come
from ``CallIdentity.typed``; nothing is defaulted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from geko.specialize.calls import UNKNOWN, CallIdentity
from geko.specialize.recipe import Refused

__all__ = ["Refused", "call_facts", "check_stock", "narrowing", "EXPRESSIBLE"]

DTYPES = {"bf16": 7, "f16": 4, "f32": 0}  # TensileLite DataType enum values
ACTIVATIONS = {"none": "none", "gelu": "gelu", "relu": "relu", "silu": "silu",
               "swish": "silu", "clamp": "clamp", "sigmoid": "sigmoid"}  # logged name -> TensileLite
GRADIENT_ACTIVATIONS = {"dgelu", "drelu", "bgrada", "bgradb"}
BIAS_HAS_M = {1, 3}  # UseBias / UseScaleAlphaVec: 1 = M, 2 = N, 3 = both
STOCK_FORWARD_DOMAIN = {"Gradient": False, "UseE": False, "OutputAmaxD": False, "UseScaleAB": "",
                        "UseScaleCD": False, "Sparse": 0, "UseGateResidual": False,
                        "GroupedGemm": False, "FusedGemmA2A": False, "MXBlockA": 0, "MXBlockB": 0,
                        "SwizzleTensorA": False, "SwizzleTensorB": False}
GENERATION_FIELDS = ("a_type", "b_type", "c_type", "d_type", "compute_type", "M", "N", "K",
                     "batch_count", "transA", "transB", "lda", "ldb", "ldc", "ldd",
                     "stride_a", "stride_b", "stride_c", "stride_d")
ORDINARY_ROUTE = "ordinary matmul API"
# Runtime route proven by the run metadata: no TunableOp, tuning override or pinned solution index,
# so the deployed library's heuristic, which the bespoke solution joins, selects the call's solution.
HEURISTIC_ROUTE = "hipBLASLt heuristic"
ASSERT_KEYS = ("AssertFree0ElementMultiple", "AssertFree1ElementMultiple",
               "AssertSummationElementMultiple")
# --exclude-controls may name these request facts; AssertSizeEqual never.
EXPRESSIBLE = ("UseBeta", "ActivationType", "UseBias", "UseScaleAlphaVec") + ASSERT_KEYS
POINTER_MODE_NAMES = {0: "host", 1: "device"}


def manifest_type(logged: str) -> str:
    """hipBLASLt type name in calls.csv spelling: bf16_r -> bf16, c_f32_r -> f32."""
    name = logged[2:] if logged.startswith("c_") else logged
    return name[:-2] if name.endswith("_r") else name


def proven(route: Optional[Mapping[str, Any]], fact: str) -> bool:
    """A route fact counts only when the route-facts entry lists evidence for it."""
    return bool(route) and fact in route and bool((route.get("evidence") or {}).get(fact))


def _missing(field: str) -> Refused:
    return Refused("missing-generation-fact", f"missing generation fact: {field}")


def identity_facts(identity: CallIdentity, bias_axis: Any = UNKNOWN) -> Dict[str, Any]:
    """Call facts from the logged identity; unknown narrowing facts stay UNKNOWN."""
    for field in GENERATION_FIELDS:
        if identity.typed(field) is UNKNOWN:
            raise _missing(field)
    call: Dict[str, Any] = {}
    for col in ("transA", "transB"):
        value = str(identity[col])
        if value not in ("N", "T"):
            raise Refused("transpose", f"{col} {value!r} is not N|T")
        call[col] = value
    for field in ("gradient", "use_e"):
        if identity.typed(field) is True:
            name = "gradient" if field == "gradient" else "aux"
            raise Refused("extra-output", f"{name} output ({field}: {identity[field]}) is outside the domain "
                          "(aux/gradient/amax/scaled-output specialization is refused)")
    for field in ("scaleA", "scaleB", "scaleC", "scaleD"):
        value = identity.typed(field)
        if value is not UNKNOWN and value not in (0, False):
            raise Refused("extra-output", f"{field}={identity[field]}: scaled input/output is outside the domain")
    types = {k: manifest_type(str(identity[k])) for k in ("a_type", "b_type", "c_type", "d_type", "compute_type")}
    bad = {k: v for k, v in types.items() if v not in DTYPES}
    if bad:
        raise Refused("dtype", f"types {bad} are outside the domain {sorted(DTYPES)}")
    if types["c_type"] != types["d_type"]:
        raise Refused("dtype", f"c_type {types['c_type']} != d_type {types['d_type']}")
    call.update(types)
    for field, key in (("M", "M"), ("N", "N"), ("K", "K"), ("batch_count", "batch"),
                       ("lda", "lda"), ("ldb", "ldb"), ("ldc", "ldc"), ("ldd", "ldd")):
        call[key] = int(identity.typed(field))

    act = identity.typed("activation_type")
    bias_vector = identity.typed("bias_vector")
    if bias_vector is UNKNOWN:
        bias = UNKNOWN
    elif not bias_vector:
        bias = "none"
    else:
        source = identity.typed("bias_source")
        if source is not UNKNOWN and str(source).lower() != "d":
            raise Refused("bias-axis", f"bias source {identity['bias_source']}: only D-source M-axis bias "
                          "is in the domain")
        bias = bias_axis
    if act is not UNKNOWN:
        act = str(act).strip().lower()
        if act in GRADIENT_ACTIVATIONS:
            raise Refused("gradient", f"activation {act!r} is a gradient epilogue (refused)")
        if act == "sigmoid" and bias_vector is True:
            raise Refused("sigmoid-bias", "sigmoid with bias is never in the domain")
        if act not in ACTIVATIONS:
            raise Refused("unsupported-activation",
                          f"activation {act!r} is not one of {sorted(ACTIVATIONS)} (never computed as another)")
    if bias not in ("none", "M", UNKNOWN):
        raise Refused("bias-axis", f"bias axis {bias!r}: only M-axis bias is in the domain (accepted values: "
                      "M; N or both axes are refused)")
    bias_type = UNKNOWN
    if bias == "M" or (bias is UNKNOWN and bias_vector is True):
        logged = identity.typed("bias_type")
        if logged is not UNKNOWN:
            bias_type = manifest_type(str(logged))
            if bias_type not in DTYPES:
                raise Refused("dtype", f"bias_type {bias_type!r} is outside the domain {sorted(DTYPES)}")
    sav = identity.typed("scaleAlpha_vector")
    pointer = identity.typed("pointer_mode")
    args = [identity.typed(f) for f in ("activation_arg0", "activation_arg1")]
    call.update({
        "alpha": identity.typed("alpha"), "beta": identity.typed("beta"),
        "pointer_raw": pointer,
        "pointer_mode": POINTER_MODE_NAMES.get(pointer, "unknown") if pointer is not UNKNOWN else "unknown",
        "bias": bias, "bias_type": bias_type, "bias_on": bias_vector,
        "activation": ACTIVATIONS[act] if act is not UNKNOWN else UNKNOWN,
        "act_args": args if act == "clamp" else [],
        "act_args_known": UNKNOWN not in args,
        "sav": UNKNOWN if sav is UNKNOWN else int(bool(sav)),
        "extra_outputs": "none", "api_path": "gemm",
    })
    return call


def route_facts(route: Optional[Mapping[str, Any]]) -> Any:
    """Check the route facts of one call; returns the proven bias axis (or UNKNOWN).

    Without route facts (``--route-facts`` omitted or no entry with evidence for the call) the
    route is unknown and the call is refused; a value outside the documented strings is refused
    with the accepted values, never read as a default."""
    if not proven(route, "call_route") or route.get("call_route") is None:
        raise Refused("missing-generation-fact", "missing generation fact: route (no route-facts entry with "
                      "evidence for this call; the trace alone does not prove the API route, see --route-facts)")
    value = str(route["call_route"])
    if value != ORDINARY_ROUTE:
        low = value.lower()
        accepted = f"; accepted call_route: {ORDINARY_ROUTE!r}"
        if "grouped" in low:
            raise Refused("grouped", f"grouped GEMM route ({value}) is outside the domain{accepted}")
        if "device" in low:
            raise Refused("device-args", f"device args route ({value}) is outside the domain{accepted}")
        if "rocroller" in low:
            raise Refused("rocroller", f"non-TensileLite route (RocRoller): {value}{accepted}")
        raise Refused("route", f"non-ordinary route ({value}); only the ordinary matmul API is specialized"
                      f"{accepted}")
    if not proven(route, "runtime_route") or route.get("runtime_route") != HEURISTIC_ROUTE:
        raise Refused("runtime-route-unproven", f"runtime route {route.get('runtime_route')!r} is not proven "
                      f"{HEURISTIC_ROUTE!r} (no TunableOp, tuning override or pinned solution index, shown "
                      "by the run metadata of both traces; see geko route-facts)")
    amax = route.get("output_amax")
    if amax is not None and not isinstance(amax, bool):
        raise Refused("route-fact-value", f"output_amax {amax!r} is not an accepted value (accepted: false = "
                      "no amax output; true = amax output, which is refused)")
    if not proven(route, "output_amax") or amax is not False:
        raise Refused("extra-output", f"amax output {amax!r} (unproven or true) is outside the domain; "
                      "only output_amax false with evidence is specialized")
    return route["bias_axis"] if proven(route, "bias_axis") and route.get("bias_axis") is not None else UNKNOWN


def call_facts(identity: CallIdentity, route: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Generation facts and domain of one call, or Refused with the reason."""
    for field in GENERATION_FIELDS:
        if identity.typed(field) is UNKNOWN:
            raise _missing(field)
    return identity_facts(identity, route_facts(route))


def _check_stock(call: Dict[str, Any], stock_pt: Dict[str, Any], batched: bool, keep_beta: bool) -> None:
    def mismatch(what):
        raise Refused("stock-mismatch", f"the stock recipe cannot compute this call: {what}")

    outside = {k: stock_pt[k] for k, forward in STOCK_FORWARD_DOMAIN.items()
               if k in stock_pt and stock_pt[k] != forward}
    if stock_pt.get("UseBias", 0) and stock_pt.get("BiasSrc", "D") != "D":
        outside["BiasSrc"] = stock_pt["BiasSrc"]
    if outside:
        raise Refused("stock-out-of-domain", f"the stock recipe is not a dense forward GEMM: {outside}")
    for tc in ("A", "B"):
        if stock_pt[f"Transpose{tc}"] != (call[f"trans{tc}"] == "T"):
            mismatch(f"Transpose{tc}={stock_pt[f'Transpose{tc}']} vs trans{tc}={call[f'trans{tc}']}")
    hpa = DTYPES[call["compute_type"]] != DTYPES[call["a_type"]]
    types = {"DataType": (stock_pt["DataType"], DTYPES[call["a_type"]]),
             "DataTypeA": (stock_pt.get("DataTypeA", stock_pt["DataType"]), DTYPES[call["a_type"]]),
             "DataTypeB": (stock_pt.get("DataTypeB", stock_pt["DataType"]), DTYPES[call["b_type"]]),
             "MacDataTypeA": (stock_pt.get("MacDataTypeA", stock_pt["DataType"]), DTYPES[call["a_type"]]),
             "MacDataTypeB": (stock_pt.get("MacDataTypeB", stock_pt["DataType"]), DTYPES[call["b_type"]]),
             "DestDataType": (stock_pt["DestDataType"], DTYPES[call["d_type"]]),
             "ComputeDataType": (stock_pt["ComputeDataType"], DTYPES[call["compute_type"]]),
             "F32XdlMathOp": (stock_pt.get("F32XdlMathOp", 0), 0),
             "HighPrecisionAccumulate": (bool(stock_pt.get("HighPrecisionAccumulate", False)), hpa)}
    wrong = {k: v for k, v in types.items() if v[0] != v[1]}
    if wrong:
        mac = (f"; the recipe's DataType / MAC input type (MacDataTypeA/B, default DataType {stock_pt['DataType']}) "
               "is not the call's A/B type") \
            if {"DataType", "MacDataTypeA", "MacDataTypeB"} & set(wrong) else ""
        raise Refused("type-mismatch", f"type-mismatch: stock vs call types differ: {wrong} (stock, call){mac}")
    if call["batch"] > 1 and not batched:
        mismatch("batch > 1 on a non-batched recipe")
    if call["beta"] is not UNKNOWN and call["beta"] != 0.0 and not stock_pt.get("UseBeta", True):
        mismatch("beta != 0 on a UseBeta: false recipe")
    stock_act = stock_pt.get("ActivationType", "none") if stock_pt.get("Activation") else "none"
    if call["activation"] not in ("none", UNKNOWN) and stock_act not in ("all", "hipblaslt_all", call["activation"]):
        mismatch(f"activation {call['activation']} on ActivationType {stock_act}")
    if call["bias"] == "M":
        if stock_pt.get("UseBias", 0) not in BIAS_HAS_M:
            mismatch(f"M-axis bias on UseBias={stock_pt.get('UseBias', 0)}")
        if call["bias_type"] is not UNKNOWN and DTYPES[call["bias_type"]] not in stock_pt.get("BiasDataTypeList", []):
            mismatch(f"bias_type {call['bias_type']} not in BiasDataTypeList {stock_pt.get('BiasDataTypeList')}")
    if call["sav"] == 1 and stock_pt.get("UseScaleAlphaVec", 0) not in BIAS_HAS_M:
        mismatch(f"scale-alpha vector on UseScaleAlphaVec={stock_pt.get('UseScaleAlphaVec', 0)}")
    # A non-host pointer mode passes alpha as a device vector of M values (the SAV
    # path); a recipe without UseScaleAlphaVec cannot serve it.
    if call["pointer_raw"] not in (0, UNKNOWN) and stock_pt.get("UseScaleAlphaVec", 0) not in BIAS_HAS_M:
        raise Refused("pointer-mode-sav", "recipe cannot serve device pointer mode (no SAV): pointer_mode "
                      f"{call['pointer_raw']}, stock UseScaleAlphaVec={stock_pt.get('UseScaleAlphaVec', 0)}")
    if keep_beta and not stock_pt.get("UseBeta", True):
        raise Refused("keep-beta", "--keep-beta needs a beta-on recipe; the stock ProblemType "
                      "has UseBeta: false")


def check_stock(call: CallIdentity | Dict[str, Any], stock_pt: Dict[str, Any], batched: bool = True,
                keep_beta: bool = False) -> None:
    """Refuse a call the fixed stock recipe cannot compute: types, transposes,
    epilogue support, out-of-domain recipes; with ``keep_beta`` a UseBeta: false recipe."""
    if isinstance(call, CallIdentity):
        call = identity_facts(call)
    _check_stock(call, stock_pt, batched, keep_beta)


def _pointer_reason(call: Dict[str, Any]) -> str:
    raw = call["pointer_raw"]
    mode = "unknown (not logged)" if raw is UNKNOWN else f"{raw} ({call['pointer_mode']})"
    return f"pointer mode {mode} is not host (0); pointer-dependent narrowing needs host pointer mode"


def narrowing(call: Dict[str, Any], stock_pt: Dict[str, Any], keep_beta: bool,
              excluded: frozenset = frozenset()) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """ProblemType changes the proven facts make true, and the facts withheld with reasons."""
    req: Dict[str, Any] = {}
    notes: List[Dict[str, Any]] = []

    def withhold(fact, reason):
        notes.append({"fact": fact, "reason": reason})

    host = call["pointer_raw"] == 0
    if stock_pt.get("UseBeta", True):
        if not host:
            withhold("UseBeta", _pointer_reason(call))
        elif call["beta"] is UNKNOWN:
            withhold("UseBeta", "beta not logged; beta stays general")
        elif call["beta"] == 0.0:
            if "UseBeta" in excluded:
                withhold("UseBeta", "excluded by --exclude-controls")
            else:
                req["UseBeta"] = bool(keep_beta)

    stock_act = stock_pt.get("ActivationType", "none") if stock_pt.get("Activation") else "none"
    act = call["activation"]
    if act is UNKNOWN:
        if stock_act != "none" or stock_pt.get("Activation"):
            withhold("ActivationType", "activation_type not logged; activation stays general")
    elif act == "none":
        if stock_act != "none" or stock_pt.get("Activation"):
            if "ActivationType" in excluded:
                withhold("ActivationType", "excluded by --exclude-controls")
            else:
                req.update(Activation=False, ActivationType="none")
    elif stock_act in ("all", "hipblaslt_all"):
        if act == "clamp" and not call["act_args_known"]:
            withhold("ActivationType", "clamp activation argument not logged; a single clamp activation "
                     "needs both arguments, activation stays general")
        elif "ActivationType" in excluded:
            withhold("ActivationType", "excluded by --exclude-controls")
        else:
            req.update(Activation=True, ActivationType=act)

    bias = call["bias"]
    if bias is UNKNOWN:
        if stock_pt.get("UseBias", 0) != 0:
            if call["bias_on"] is UNKNOWN:
                withhold("UseBias", "bias_vector not logged; bias stays general")
            else:
                withhold("BiasDataTypeList", "bias axis not proven by route facts; bias stays general")
    elif bias == "none":
        if stock_pt.get("UseBias", 0) != 0:
            if "UseBias" in excluded:
                withhold("UseBias", "excluded by --exclude-controls")
            else:
                req.update(UseBias=0, BiasDataTypeList=[])
    elif call["bias_type"] is UNKNOWN:
        withhold("BiasDataTypeList", "bias_type not logged; bias types stay general")
    elif stock_pt.get("BiasDataTypeList") != [DTYPES[call["bias_type"]]]:
        if "UseBias" in excluded:
            withhold("UseBias", "excluded by --exclude-controls")
        else:
            req["BiasDataTypeList"] = [DTYPES[call["bias_type"]]]

    # TensileLite rejects a single sigmoid activation together with bias; when bias stays
    # general (unknown, unproven or excluded), the single sigmoid stays general too.
    if req.get("ActivationType") == "sigmoid" and req.get("UseBias", stock_pt.get("UseBias", 0)) != 0:
        del req["ActivationType"], req["Activation"]
        withhold("ActivationType", "bias unknown or kept general; sigmoid+UseBias unsupported, so the "
                 "single sigmoid activation is not applied")

    if stock_pt.get("UseScaleAlphaVec", 0) != 0:
        if not host:
            withhold("UseScaleAlphaVec", _pointer_reason(call))
        elif call["sav"] is UNKNOWN:
            withhold("UseScaleAlphaVec", "scaleAlpha_vector not logged; SAV stays general")
        elif call["sav"] == 0:
            if "UseScaleAlphaVec" in excluded:
                withhold("UseScaleAlphaVec", "excluded by --exclude-controls")
            else:
                req["UseScaleAlphaVec"] = 0
    return req, notes

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Translate recorded convolution workloads into rocKE dispatcher requests.

Accepts the kernel team's ``bench_cases_conv.json`` and hipDNN frontend graph
JSON, either in a directory or in a dnn-benchmarking workload tarball. Dimensions
and convolution attributes come from the records, never from filenames.

``--out`` contains deduplicated ConvGroupedRequest mappings for the integration's
2D, channels-last, groups=1, FP16/BF16 forward cross-correlation contract.
``--report`` (default: <out-stem>.report.json) retains EVERY input, its original
attributes, exclusions, and any representable dispatcher request. Exclusions are
integration scope decisions, not claims about the underlying rocKE family. Kernel
support is pending until the actual dispatcher's validity predicate is evaluated.

Example::

    python mine_conv_shapes.py --rocke-cases bench_cases_conv.json \\
        --graphs conv.tar.gz --arch gfx950 --out conv_requests.json

Use ``--source-uri LOCAL_PATH URI`` to record a pinned public repository or DVC
object URI alongside the input's SHA256. No download, compilation or GPU launch
is performed by this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path


class MiningError(ValueError):
    """The source cannot be interpreted without guessing its semantics."""


_DTYPES = {
    "half": "fp16",
    "fp16": "fp16",
    "float16": "fp16",
    "torch.float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "float": "fp32",
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "int8": "int8",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}
_LAYOUTS = {"NHWC", "NCHW", "NDHWC", "NCDHW", "NLC", "NCL"}
_OPS = {"conv1d", "conv2d", "conv3d", "conv2d_transpose", "dcnv3"}
_CONV_MODES = {"CROSS_CORRELATION", "CONVOLUTION"}
_CONV_NODES = {
    "ConvolutionFwdAttributes",
    "ConvolutionBwdAttributes",
    "ConvolutionWrwAttributes",
}
_CASE_FIELDS = {
    "N",
    "Cin",
    "Cout",
    "C",
    "K",
    "D",
    "Di",
    "H",
    "Hi",
    "W",
    "Wi",
    "Kd",
    "Z",
    "Kh",
    "Y",
    "Kw",
    "X",
    "L",
    "Kl",
    "stride",
    "strides",
    "pad",
    "pads_before",
    "pads_after",
    "dilation",
    "dilations",
    "groups",
    "dtype",
    "layout",
    "op",
    "conv_mode",
    "output_spatial",
    "case_id",
    "model",
    "variant",
    "domain",
    "owner_story",
    "capability",
    "used_by",
    "note",
    "suite",
    "purpose",
    "priority",
    "pre_op",
    "causal",
    "execution",
    "cache_semantics",
    "composition_id",
    "branch",
    "composition_op",
    "composition_timing",
    "runtime_gate",
    "output_padding",
    "channels_per_group",
    "deform_groups",
    "modulation_mask_tensor",
    "offset_tensor",
    "offset_scale",
    "remove_center",
}


def _category(value, allowed, label: str, source: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise MiningError(f"{source}: unknown {label} {value!r}")
    return value


def _dtype(value, source: str) -> str:
    if not isinstance(value, str) or value.lower() not in _DTYPES:
        raise MiningError(f"{source}: unknown dtype {value!r}")
    return _DTYPES[value.lower()]


def _int(value, label: str, source: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise MiningError(
            f"{source}: {label} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _vector(value, rank: int, label: str, source: str, minimum: int = 1) -> list[int]:
    values = [value] * rank if type(value) is int else value
    if not isinstance(values, list) or len(values) != rank:
        raise MiningError(
            f"{source}: {label} must have {rank} components, got {value!r}"
        )
    return [_int(v, label, source, minimum) for v in values]


def _alias(raw: dict, *names: str, default=None):
    values = [raw[n] for n in names if n in raw]
    if values and any(v != values[0] for v in values[1:]):
        raise MiningError(f"conflicting aliases {names}: {values}")
    return values[0] if values else default


def _vector_alias(raw, names, rank, source, *, default, minimum=1):
    values = [_vector(raw[n], rank, n, source, minimum) for n in names if n in raw]
    if values and any(v != values[0] for v in values[1:]):
        raise MiningError(f"{source}: conflicting aliases {names}: {values}")
    return values[0] if values else _vector(default, rank, names[0], source, minimum)


def _provenance(
    source: str, path: str, blob: bytes, uri: str | None, raw: dict, **extra
) -> dict:
    return {
        "source": source,
        "path": path,
        "uri": uri or path,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "case_id": raw.get("case_id"),
        "graph": raw.get("name"),
        "model": raw.get("model", (raw.get("metadata") or {}).get("model")),
        "priority": raw.get("priority"),
        "suite": raw.get("suite"),
        "attributes": copy.deepcopy(raw),
        **extra,
    }


def _result(
    provenance: dict, reasons: list[str], request: dict | None, *, forward=True
) -> dict:
    return {
        "status": (
            "non_forward"
            if not forward
            else "outside_contract" if reasons else "request"
        ),
        "reasons": reasons,
        "provenance": provenance,
        # This is an exact request for the recorded convolution, including groups
        # and depth where representable. It does not assert family support.
        "oracle_request": request,
        "kernel_support": "pending" if request is not None else "not_representable",
    }


def _make_request(
    *, n, c, k, spatial, kernel, stride, padding, dilation, groups, dtype, layout, arch
):
    if len(spatial) not in (2, 3):
        return None
    # The dispatcher calls its channels-last layout NHWC even when depth is
    # supplied. The original NDHWC/NCDHW spelling remains in the provenance.
    layout = {"NDHWC": "NHWC", "NCDHW": "NCHW"}.get(layout, layout)
    request = {
        "N": n,
        "C": c,
        "K": k,
        "Hi": spatial[-2],
        "Wi": spatial[-1],
        "Y": kernel[-2],
        "X": kernel[-1],
        "G": groups,
        "stride_h": stride[-2],
        "stride_w": stride[-1],
        "pad_h": padding[-2],
        "pad_w": padding[-1],
        "dilation_h": dilation[-2],
        "dilation_w": dilation[-1],
        "dtype": dtype,
        "layout": layout,
        "direction": "fwd",
        "arch": arch,
    }
    if len(spatial) == 3:
        request.update(
            Di=spatial[0],
            Z=kernel[0],
            stride_d=stride[0],
            pad_d=padding[0],
            dilation_d=dilation[0],
        )
    return request


def _scope_reasons(rank, groups, dtype, layout, mode, pre, post) -> list[str]:
    reasons = []
    if rank != 2:
        reasons.append(
            f"integration contract covers 2D convolution, recorded spatial rank is {rank}"
        )
    if groups != 1:
        reasons.append(
            f"integration contract covers groups=1, recorded groups={groups}"
        )
    if dtype not in ("fp16", "bf16"):
        reasons.append(f"integration contract covers fp16/bf16, recorded dtype={dtype}")
    if layout not in ("NHWC", "NDHWC", "NLC"):
        reasons.append(
            f"integration contract requires channels-last storage, recorded layout={layout}"
        )
    if mode != "CROSS_CORRELATION":
        reasons.append(
            f"integration contract requires CROSS_CORRELATION, recorded mode={mode}"
        )
    if pre != post:
        reasons.append(
            "integration contract and dispatcher request require symmetric pre/post padding"
        )
    return reasons


def _output_spatial(spatial, kernel, stride, pre, post, dilation, source) -> list[int]:
    output = [
        (i + p + q - d * (f - 1) - 1) // s + 1
        for i, f, s, p, q, d in zip(spatial, kernel, stride, pre, post, dilation)
    ]
    if any(v <= 0 for v in output):
        raise MiningError(
            f"{source}: convolution has nonpositive output dimensions {output}"
        )
    return output


def mine_case(raw: dict, provenance: dict, arch: str) -> dict:
    source = f"{provenance['path']}[{provenance.get('row', '?')}]"
    op = _category(raw.get("op"), _OPS, "convolution op", source)
    layout = _category(raw.get("layout"), _LAYOUTS, "layout", source)
    dtype = _dtype(raw.get("dtype"), source)
    mode = _category(
        raw.get("conv_mode", "CROSS_CORRELATION"), _CONV_MODES, "conv_mode", source
    )
    if op in ("conv2d_transpose", "dcnv3"):
        return _result(
            provenance, [f"integration contract does not implement {op}"], None
        )
    rank = {"conv1d": 1, "conv2d": 2, "conv3d": 3}[op]
    layouts = {1: {"NCL", "NLC"}, 2: {"NCHW", "NHWC"}, 3: {"NCDHW", "NDHWC"}}
    if layout not in layouts[rank]:
        raise MiningError(f"{source}: {layout} is inconsistent with {op}")
    n = _int(raw.get("N"), "N", source)
    c = _int(_alias(raw, "Cin", "C"), "Cin", source)
    k = _int(_alias(raw, "Cout", "K"), "Cout", source)
    groups = _int(raw.get("groups", 1), "groups", source)
    if c % groups or k % groups:
        raise MiningError(f"{source}: groups={groups} must divide Cin={c} and Cout={k}")
    if rank == 1:
        spatial, kernel = [raw.get("L")], [raw.get("Kl")]
    else:
        spatial = [_alias(raw, "H", "Hi"), _alias(raw, "W", "Wi")]
        kernel = [_alias(raw, "Kh", "Y"), _alias(raw, "Kw", "X")]
        if rank == 3:
            spatial.insert(0, _alias(raw, "D", "Di"))
            kernel.insert(0, _alias(raw, "Kd", "Z"))
    spatial = _vector(spatial, rank, "input spatial dimensions", source)
    kernel = _vector(kernel, rank, "filter spatial dimensions", source)
    stride = _vector_alias(raw, ("strides", "stride"), rank, source, default=1)
    pre = _vector_alias(raw, ("pads_before", "pad"), rank, source, default=0, minimum=0)
    post = _vector(raw.get("pads_after", pre), rank, "post_padding", source, 0)
    dilation = _vector_alias(raw, ("dilations", "dilation"), rank, source, default=1)
    output = _output_spatial(spatial, kernel, stride, pre, post, dilation, source)
    if (
        "output_spatial" in raw
        and _vector(raw["output_spatial"], rank, "output_spatial", source) != output
    ):
        raise MiningError(
            f"{source}: recorded output_spatial disagrees with convolution attributes"
        )
    reasons = _scope_reasons(rank, groups, dtype, layout, mode, pre, post)
    semantic_reasons = []
    if raw.get("causal"):
        semantic_reasons.append(
            "causal convolution requires explicit padding/cache semantics"
        )
    if raw.get("execution") not in (None, "full_clip") or raw.get("cache_semantics"):
        semantic_reasons.append(
            "stateful/streaming execution is outside the integration contract"
        )
    if raw.get("composition_id") or raw.get("composition_op"):
        semantic_reasons.append(
            "multi-operation composition is outside the integration contract"
        )
    pre_op = raw.get("pre_op")
    if pre_op is not None and not (
        isinstance(pre_op, dict)
        and pre_op.get("type") == "external_upsample"
        and pre_op.get("benchmarked") is False
    ):
        semantic_reasons.append(
            "recorded pre_op is not an explicitly excluded external upsample"
        )
    unknown = sorted(raw.keys() - _CASE_FIELDS)
    if unknown:
        semantic_reasons.append(f"uninterpreted source attributes: {unknown}")
    if raw.get("runtime_gate"):
        semantic_reasons.append(
            "recorded runtime gate is outside the integration contract"
        )
    reasons.extend(semantic_reasons)
    request = None
    if pre == post and mode == "CROSS_CORRELATION" and not semantic_reasons:
        request = _make_request(
            n=n,
            c=c,
            k=k,
            spatial=spatial,
            kernel=kernel,
            stride=stride,
            padding=pre,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            layout=layout,
            arch=arch,
        )
    return _result(provenance, reasons, request)


def _tensor_layout(dims: list[int], strides: list[int]) -> str:
    rank = len(dims)
    if rank not in (3, 4, 5):
        return "unsupported_rank"
    for order, labels in (
        ([0, *range(2, rank), 1], {3: "NLC", 4: "NHWC", 5: "NDHWC"}),
        (list(range(rank)), {3: "NCL", 4: "NCHW", 5: "NCDHW"}),
    ):
        expected = [0] * rank
        stride = 1
        for axis in reversed(order):
            expected[axis] = stride
            stride *= dims[axis]
        # A singleton dimension's stride cannot change any accessed address.
        if all(
            d == 1 or actual == wanted
            for d, actual, wanted in zip(dims, strides, expected)
        ):
            return labels[rank]
    return "non_dense"


def mine_graph(raw: dict, provenance: dict, arch: str) -> dict:
    source = provenance["path"]
    override_shapes = raw.get("is_override_shape_enabled", False)
    if type(override_shapes) is not bool:
        raise MiningError(
            f"{source}: is_override_shape_enabled must be a boolean, got {override_shapes!r}"
        )
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise MiningError(f"{source}: expected graph nodes array")
    types = [node.get("type") for node in nodes]
    for kind in types:
        if (
            isinstance(kind, str)
            and kind.startswith("Convolution")
            and kind not in _CONV_NODES
        ):
            raise MiningError(f"{source}: unknown convolution node type {kind!r}")
    forwards = [n for n in nodes if n.get("type") == "ConvolutionFwdAttributes"]
    if not forwards:
        return _result(
            provenance,
            ["graph contains no forward convolution node"],
            None,
            forward=False,
        )
    if len(nodes) != 1:
        return _result(
            provenance,
            [
                f"integration contract requires a single convolution node, recorded nodes={types}"
            ],
            None,
        )
    node = forwards[0]
    attrs = node.get("parameters")
    if not isinstance(attrs, dict):
        raise MiningError(f"{source}: expected convolution parameters object")
    mode = _category(attrs.get("conv_mode"), _CONV_MODES, "conv_mode", source)
    tensors = raw.get("tensors")
    if not isinstance(tensors, list) or any(
        not isinstance(t, dict) or "uid" not in t for t in tensors
    ):
        raise MiningError(f"{source}: expected tensors with unique uid values")
    by_uid = {t["uid"]: t for t in tensors}
    if len(by_uid) != len(tensors):
        raise MiningError(f"{source}: duplicate tensor uid")
    try:
        x = by_uid[node["inputs"]["x_tensor_uid"]]
        w = by_uid[node["inputs"]["w_tensor_uid"]]
        y = by_uid[node["outputs"]["y_tensor_uid"]]
    except (KeyError, TypeError) as exc:
        raise MiningError(
            f"{source}: unresolved convolution tensor binding: {exc}"
        ) from exc
    resolved = []
    for tensor in (x, w, y):
        dims = tensor.get("dims")
        if not isinstance(dims, list) or len(dims) < 3:
            raise MiningError(
                f"{source}: tensor dims must have batch, channel and spatial axes"
            )
        dims = _vector(dims, len(dims), "tensor dims", source)
        strides = _vector(tensor.get("strides"), len(dims), "tensor strides", source, 0)
        dtype = _dtype(tensor.get("data_type", raw.get("io_data_type")), source)
        resolved.append((dims, strides, dtype, _tensor_layout(dims, strides)))
    xd, _, dtype, layout = resolved[0]
    wd, _, wdtype, wlayout = resolved[1]
    yd, _, ydtype, ylayout = resolved[2]
    rank = len(xd) - 2
    if len(wd) != len(xd) or len(yd) != len(xd):
        raise MiningError(f"{source}: input, filter and output ranks disagree")
    if xd[1] % wd[1] or wd[0] % (xd[1] // wd[1]):
        raise MiningError(f"{source}: invalid grouped convolution channel dimensions")
    groups = xd[1] // wd[1]
    pre = _vector(attrs.get("pre_padding"), rank, "pre_padding", source, 0)
    post = _vector(attrs.get("post_padding"), rank, "post_padding", source, 0)
    stride = _vector(attrs.get("stride"), rank, "stride", source)
    dilation = _vector(attrs.get("dilation"), rank, "dilation", source)
    output = _output_spatial(xd[2:], wd[2:], stride, pre, post, dilation, source)
    if yd != [xd[0], wd[0], *output]:
        raise MiningError(
            f"{source}: output tensor dims disagree with convolution attributes"
        )
    reasons = _scope_reasons(rank, groups, dtype, layout, mode, pre, post)
    exact = True
    if override_shapes:
        reasons.append(
            "integration contract does not support execution-time shape overrides"
        )
        exact = False
    if set(node.get("inputs", {})) != {"x_tensor_uid", "w_tensor_uid"} or set(
        node.get("outputs", {})
    ) != {"y_tensor_uid"}:
        reasons.append("uninterpreted convolution tensor bindings")
        exact = False
    if len(tensors) != 3 or len({x["uid"], w["uid"], y["uid"]}) != 3:
        reasons.append(
            "integration contract requires exactly three distinct bound tensors"
        )
        exact = False
    if dtype != wdtype or dtype != ydtype:
        reasons.append(
            "integration contract and dispatcher require a uniform input/filter/output dtype"
        )
        exact = False
    if (
        layout != wlayout
        or layout != ylayout
        or layout in ("non_dense", "unsupported_rank")
    ):
        reasons.append(
            f"tensor layouts cannot be represented by one dispatcher layout: {layout}/{wlayout}/{ylayout}"
        )
        exact = False
    compute = _dtype(
        node.get("compute_data_type", raw.get("compute_data_type")), source
    )
    if compute != "fp32":
        reasons.append(
            f"integration contract requires fp32 accumulation, recorded compute dtype={compute}"
        )
        exact = False
    if any(t.get("virtual", False) is not False for t in (x, w, y)):
        reasons.append(
            "integration contract requires non-virtual input/filter/output tensors"
        )
        exact = False
    unknown = sorted(
        attrs.keys()
        - {"conv_mode", "pre_padding", "post_padding", "stride", "dilation"}
    )
    if unknown:
        reasons.append(f"uninterpreted convolution attributes: {unknown}")
        exact = False
    for tensor in (x, w, y):
        unknown = sorted(
            tensor.keys() - {"uid", "name", "dims", "strides", "data_type", "virtual"}
        )
        if unknown:
            reasons.append(
                f"uninterpreted tensor attributes for uid={tensor['uid']}: {unknown}"
            )
            exact = False
    request = None
    if exact and mode == "CROSS_CORRELATION" and pre == post:
        request = _make_request(
            n=xd[0],
            c=xd[1],
            k=wd[0],
            spatial=xd[2:],
            kernel=wd[2:],
            stride=stride,
            padding=pre,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
            layout=layout,
            arch=arch,
        )
    return _result(provenance, reasons, request)


def _json(blob: bytes, source: str):
    try:
        return json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiningError(f"{source}: invalid JSON: {exc}") from exc


def from_rocke_cases(path: Path, arch="gfx950", source_uri=None) -> list[dict]:
    blob = path.read_bytes()
    raw = _json(blob, str(path))
    if not isinstance(raw, list) or any(not isinstance(r, dict) for r in raw):
        raise MiningError(f"{path}: expected an array of convolution case objects")
    return [
        mine_case(
            row,
            _provenance("rocke_bench", str(path), blob, source_uri, row, row=i),
            arch,
        )
        for i, row in enumerate(raw)
    ]


def from_graph_corpus(path: Path, arch="gfx950", source_uri=None) -> list[dict]:
    records = []
    if path.is_dir():
        blobs = [
            (p.relative_to(path).as_posix(), p.read_bytes())
            for p in sorted(path.rglob("*.json"))
        ]
        archive_digest = None
    elif tarfile.is_tarfile(path):
        archive_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with tarfile.open(path, "r:*") as archive:
            blobs = [
                (m.name, archive.extractfile(m).read())
                for m in sorted(archive.getmembers(), key=lambda m: m.name)
                if m.isfile() and m.name.endswith(".json")
            ]
    else:
        blobs = [(path.name, path.read_bytes())]
        archive_digest = None
    for name, blob in blobs:
        raw = _json(blob, name)
        if not isinstance(raw, dict):
            raise MiningError(f"{name}: expected a graph object")
        uri = (
            f"{source_uri}#{name}"
            if source_uri
            else str(path / name) if path.is_dir() else f"{path}#{name}"
        )
        provenance = _provenance(
            "graphs", name, blob, uri, raw, archive_sha256=archive_digest
        )
        records.append(mine_graph(raw, provenance, arch))
    return records


def build_corpus(records: list[dict]) -> tuple[list[dict], dict]:
    requests = {}
    for record in records:
        if record["status"] != "request":
            continue
        request = record["oracle_request"]
        if request is None:
            raise MiningError(
                f"{record['provenance']['path']}: contract request is not representable"
            )
        key = json.dumps(request, sort_keys=True)
        provenance = copy.deepcopy(record["provenance"])
        if key in requests:
            requests[key]["_provenance"].setdefault("also", []).append(provenance)
        else:
            requests[key] = {**request, "_provenance": provenance}
    counts = Counter(r["status"] for r in records)
    report = {
        "schema_version": 1,
        "meaning": "Source inventory and integration contract eligibility; kernel validity and runtime dispatch are not evaluated.",
        "counts": {
            "source_records": len(records),
            **dict(counts),
            "unique_requests": len(requests),
            "merged_request_records": counts["request"] - len(requests),
        },
        "by_source": {
            source: dict(
                Counter(
                    r["status"] for r in records if r["provenance"]["source"] == source
                )
            )
            for source in sorted({r["provenance"]["source"] for r in records})
        },
        "records": records,
    }
    return list(requests.values()), report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rocke-cases", type=Path, action="append", default=[])
    parser.add_argument("--graphs", type=Path, action="append", default=[])
    parser.add_argument(
        "--source-uri",
        nargs=2,
        action="append",
        default=[],
        metavar=("LOCAL_PATH", "URI"),
    )
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not args.rocke_cases and not args.graphs:
        parser.error("provide at least one source: --rocke-cases or --graphs")
    uris = {str(Path(path).resolve()): uri for path, uri in args.source_uri}
    sources = {str(path.resolve()) for path in [*args.rocke_cases, *args.graphs]}
    if uris.keys() - sources:
        parser.error("--source-uri LOCAL_PATH must name an input source")
    try:
        records = []
        for paths, reader in (
            (args.rocke_cases, from_rocke_cases),
            (args.graphs, from_graph_corpus),
        ):
            for path in paths:
                records.extend(reader(path, args.arch, uris.get(str(path.resolve()))))
        requests, report = build_corpus(records)
        report_path = args.report or args.out.with_name(args.out.stem + ".report.json")
        for path, payload in ((args.out, requests), (report_path, report)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(report["counts"], sort_keys=True))
        print(
            f"Kernel support pending actual dispatcher validation; inventory: {report_path}"
        )
    except (MiningError, OSError, tarfile.TarError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

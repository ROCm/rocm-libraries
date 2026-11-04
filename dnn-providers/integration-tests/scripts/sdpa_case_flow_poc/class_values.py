# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn the model's class_to_value table into the machine-readable file the discovery driver reads.

Every value is read from a typed field of the table; `note` fields are never read. A missing or
ill-typed field, or a table that misses a value of its axis, is a ModelError, never a silent default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .model import Model, ModelError, load_yaml

SCHEMA_VERSION = 2
FILE_NAME = "class-to-value.json"
INVERSE_SQRT_HEAD_DIM = "1/sqrt(head_dim_qk)"
INVERSE_SQRT_HEAD_DIM_TOKEN = "inverse_sqrt_head_dim_qk"
DIM_LETTERS = ("B", "H", "S", "D")
TENSOR_LETTERS = ("Q", "K", "V", "O")
STATS_LETTER = "Stats"
LAYOUT_AXES = ("strides_q", "strides_k", "strides_v", "strides_o", "strides_stats")
NOTE = "note"
STORED_LENGTH = "S"
LENGTH_OPERATIONS = ("divide", "add", "multiply")
OFFSET_LAYOUTS = ("prefix_sum_of_lengths", "full_slots", "prefix_sum_after_gap")


def _without_notes(value):
    if isinstance(value, dict):
        return {k: _without_notes(v) for k, v in value.items() if k != NOTE}
    if isinstance(value, list):
        return [_without_notes(v) for v in value]
    return value


def _number_list(value, where: str) -> list:
    if not isinstance(value, list) or not value:
        raise ModelError(f"class_to_value: {where} is not a non-empty list: {value!r}")
    return value


def _entry(table: dict, key: str, where: str):
    if not isinstance(table, dict):
        raise ModelError(f"class_to_value: {where} is not a mapping")
    if key not in table:
        raise ModelError(f"class_to_value: {where} has no entry '{key}'")
    return table[key]


def _covers(table: dict, model: Model, axis: str, where: str) -> None:
    if not isinstance(table, dict):
        raise ModelError(f"class_to_value: {where} is not a mapping")
    missing = [v for v in model.values[axis] if v not in table]
    if missing:
        raise ModelError(f"class_to_value: {where} has no value for {axis} {missing}")


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"class_to_value: {where} is not a number: {value!r}")
    return value


def _integer(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"class_to_value: {where} is not an integer: {value!r}")
    return value


def _format(value, model: Model, where: str) -> str:
    if value not in model.values["dtype_q"]:
        raise ModelError(f"class_to_value: {where} is not a format: {value!r}")
    return value


def _format_list(value, model: Model, where: str) -> list:
    if not isinstance(value, list) or not value:
        raise ModelError(
            f"class_to_value: {where} is not a non-empty list of formats: {value!r}"
        )
    return [_format(v, model, f"{where}[{i}]") for i, v in enumerate(value)]


def _length_term(value, where: str):
    """One per-sequence length: S, an integer, or {of: S, <divide|add|multiply>: n}."""
    if value == STORED_LENGTH:
        return value
    if isinstance(value, bool):
        raise ModelError(f"class_to_value: {where} is not a length: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ModelError(f"class_to_value: {where} is a negative length: {value!r}")
        return value
    if (
        isinstance(value, dict)
        and len(value) == 2
        and value.get("of") == STORED_LENGTH
        and any(op in value for op in LENGTH_OPERATIONS)
    ):
        op = next(op for op in LENGTH_OPERATIONS if op in value)
        n = _integer(value[op], f"{where}.{op}")
        if op == "divide" and n <= 0:
            raise ModelError(f"class_to_value: {where}.divide is not positive: {n!r}")
        return {"of": STORED_LENGTH, op: n}
    raise ModelError(
        f"class_to_value: {where} is not S, an integer or {{of: S, divide|add|multiply: n}}: {value!r}"
    )


def _scale_representative(value, where: str):
    if value == INVERSE_SQRT_HEAD_DIM_TOKEN:
        return INVERSE_SQRT_HEAD_DIM
    if isinstance(value, dict):
        if set(value) != {"numerator", "denominator"}:
            raise ModelError(
                f"class_to_value: {where} needs exactly numerator and denominator: {value!r}"
            )
        den = _number(value["denominator"], where + ".denominator")
        if den == 0:
            raise ModelError(f"class_to_value: {where} has denominator 0")
        return float(_number(value["numerator"], where + ".numerator")) / float(den)
    n = float(_number(value, where))
    if math.isnan(n):
        return "nan"
    if math.isinf(n):
        return "+inf" if n > 0 else "-inf"
    return n


def _scale_representatives(value, where: str) -> list:
    """The representatives of one scale class, primary first."""
    reps = _entry(value, "representatives", where)
    if not isinstance(reps, list) or not reps:
        raise ModelError(
            f"class_to_value: {where}.representatives is not a non-empty list: {reps!r}"
        )
    return [
        _scale_representative(r, f"{where}.representatives[{i}]")
        for i, r in enumerate(reps)
    ]


def _dim_order(value, where: str) -> list:
    order = _entry(value, "order", where)
    if not isinstance(order, list) or sorted(order) != sorted(DIM_LETTERS):
        raise ModelError(
            f"class_to_value: {where}.order must list B, H, S, D once each: {order!r}"
        )
    return list(order)


def _differs_formats(rel: dict, model: Model, where: str) -> tuple:
    """dtype_q -> K/V format for 'differs', and per-arch format substitutions."""
    when_q = _format_list(_entry(rel, "when_q", where), model, where + ".when_q")
    if len(set(when_q)) != len(when_q):
        raise ModelError(f"class_to_value: {where}.when_q names a format twice")
    k_v = _format(_entry(rel, "k_v", where), model, where + ".k_v")
    otherwise = _format(_entry(rel, "otherwise", where), model, where + ".otherwise")
    by_q = {q: k_v for q in when_q}
    for q in model.values["dtype_q"]:
        by_q.setdefault(q, otherwise)

    by_arch = {}
    arch_table = _entry(rel, "by_arch", where)
    if not isinstance(arch_table, dict):
        raise ModelError(f"class_to_value: {where}.by_arch is not a mapping")
    for arch, subs in arch_table.items():
        if not isinstance(subs, dict) or not subs:
            raise ModelError(
                f"class_to_value: {where}.by_arch.{arch} is not a non-empty mapping"
            )
        by_arch[str(arch)] = {
            _format(src, model, f"{where}.by_arch.{arch} key"): _format(
                dst, model, f"{where}.by_arch.{arch}.{src}"
            )
            for src, dst in subs.items()
        }
    return by_q, by_arch


def build(model: Model, model_dir: Path) -> dict:
    table = load_yaml(Path(model_dir) / "model.yaml").get("class_to_value")
    if not isinstance(table, dict):
        raise ModelError("model.yaml has no class_to_value table")

    size = {}
    for axis in ("batch_size", "q_heads"):
        t = _entry(table, axis, "class_to_value")
        _covers(t, model, axis, axis)
        size[axis] = {k: _integer(t[k], f"{axis}.{k}") for k in model.values[axis]}

    seq_table = _entry(table, "seq_lengths", "class_to_value")
    if not isinstance(seq_table, list):
        raise ModelError("class_to_value: seq_lengths is not a list")
    seq = []
    for i, e in enumerate(seq_table):
        where = f"seq_lengths[{i}]"
        row = {k: _entry(e, k, where) for k in ("seq_q", "seq_kv", "seq_q_vs_seq_kv")}
        for k, v in row.items():
            if v not in model.values[k]:
                raise ModelError(f"class_to_value: {where} unknown value {k}={v}")
        row["S_q"] = _integer(_entry(e, "S_q", where), where + ".S_q")
        row["S_kv"] = _integer(_entry(e, "S_kv", where), where + ".S_kv")
        seq.append(row)

    hd = _entry(table, "head_dim", "class_to_value")
    for axis in ("head_dim_qk", "head_dim_v"):
        _covers(hd, model, axis, "head_dim")
    known_dims = set(model.values["head_dim_qk"]) | set(model.values["head_dim_v"])
    head_dim = {}
    for k, v in hd.items():
        if k == NOTE:
            continue
        if k not in known_dims:
            raise ModelError(f"class_to_value: head_dim has unknown value {k}")
        head_dim[k] = _integer(v, f"head_dim.{k}")

    rel = _entry(table, "dtype_relation", "class_to_value")
    by_q, by_arch = _differs_formats(
        _entry(rel, "differs", "dtype_relation"), model, "dtype_relation.differs"
    )

    do = _entry(table, "dim_order", "class_to_value")
    for axis in ("dim_order_q", "dim_order_k", "dim_order_v", "dim_order_o"):
        _covers(do, model, axis, "dim_order")
    strides_table = _entry(table, "strides", "class_to_value")
    for axis in LAYOUT_AXES:
        _covers(strides_table, model, axis, "strides")
    dim_order = {
        k: _dim_order(do[k], f"dim_order.{k}") for k in model.values["dim_order_q"]
    }

    sv = _entry(table, "scale_value", "class_to_value")
    _covers(sv, model, "scale_value", "scale_value")
    scale = {
        k: _scale_representatives(sv[k], f"scale_value.{k}")
        for k in model.values["scale_value"]
    }

    dp = _entry(table, "dropout_probability", "class_to_value")
    _covers(dp, model, "dropout_probability", "dropout_probability")
    dropout_p = {
        k: float(_number(dp[k], f"dropout_probability.{k}"))
        for k in model.values["dropout_probability"]
    }

    rng = _entry(
        _entry(table, "dropout_rng_state", "class_to_value"),
        "caller_supplied",
        "dropout_rng_state",
    )
    where = "dropout_rng_state.caller_supplied"
    seed_offset = {
        "seed": _integer(_entry(rng, "seed", where), where + ".seed"),
        "offset": _integer(_entry(rng, "offset", where), where + ".offset"),
    }

    mp = _entry(table, "dropout_mask_pattern", "class_to_value")
    _covers(mp, model, "dropout_mask_pattern", "dropout_mask_pattern")
    mask_scale = {
        k: float(
            _number(
                _entry(mp[k], "scale", f"dropout_mask_pattern.{k}"),
                f"dropout_mask_pattern.{k}.scale",
            )
        )
        for k in model.values["dropout_mask_pattern"]
    }
    mask_cycle = {}
    for k in model.values["dropout_mask_pattern"]:
        cyc = _entry(mp[k], "cycle", f"dropout_mask_pattern.{k}")
        if not isinstance(cyc, list) or not cyc:
            raise ModelError(
                f"class_to_value: dropout_mask_pattern.{k}.cycle is not a non-empty list"
            )
        mask_cycle[k] = [
            float(_number(x, f"dropout_mask_pattern.{k}.cycle")) for x in cyc
        ]

    ps = _entry(_entry(table, "paging", "class_to_value"), "page_size", "paging")
    _covers(ps, model, "page_size", "paging.page_size")
    page_size = {}
    for k in model.values["page_size"]:
        v = ps[k]
        if v != "S_kv":
            v = _integer(v, f"paging.page_size.{k}")
        page_size[k] = v

    qt = _entry(table, "descale_and_scale_tensors", "class_to_value")
    factors = {
        key: float(
            _number(
                _entry(qt, key, "descale_and_scale_tensors"),
                f"descale_and_scale_tensors.{key}",
            )
        )
        for key in ("descale_q", "descale_k", "descale_v", "other")
    }
    descale_lists = {}
    for granularity in ("per_head", "finer"):
        where = f"descale_and_scale_tensors.{granularity}"
        vals = _entry(
            _entry(qt, granularity, "descale_and_scale_tensors"), "values", where
        )
        descale_lists[granularity] = {
            key: [
                float(_number(x, f"{where}.values.{key}"))
                for x in _number_list(
                    _entry(vals, key, where + ".values"), f"{where}.values.{key}"
                )
            ]
            for key in ("descale_q", "descale_k", "descale_v")
        }

    packing = _entry(
        _entry(table, "batch_packing", "class_to_value"),
        "ragged_tensors",
        "batch_packing",
    )
    _covers(packing, model, "batch_packing", "batch_packing.ragged_tensors")
    ragged_tensors = {}
    for k in model.values["batch_packing"]:
        letters = packing[k]
        if not isinstance(letters, list) or any(
            x not in TENSOR_LETTERS for x in letters
        ):
            raise ModelError(
                f"class_to_value: batch_packing.ragged_tensors.{k} is not a list of {TENSOR_LETTERS}"
            )
        ragged_tensors[k] = list(letters)

    stats_table = _entry(
        _entry(table, "batch_packing", "class_to_value"),
        "ragged_stats",
        "batch_packing",
    )
    _covers(stats_table, model, "stats_packing", "batch_packing.ragged_stats")
    ragged_stats = {}
    for k in model.values["stats_packing"]:
        letters = stats_table[k]
        if not isinstance(letters, list) or any(x != STATS_LETTER for x in letters):
            raise ModelError(
                f"class_to_value: batch_packing.ragged_stats.{k} is not a list of {STATS_LETTER}"
            )
        ragged_stats[k] = list(letters)

    ro = _entry(table, "ragged_offsets", "class_to_value")
    _covers(ro, model, "ragged_offsets", "ragged_offsets")
    ragged_offsets = {k: _without_notes(ro[k]) for k in model.values["ragged_offsets"]}
    for k, entry in ragged_offsets.items():
        layout = _entry(entry, "offsets", f"ragged_offsets.{k}")
        if layout not in OFFSET_LAYOUTS:
            raise ModelError(
                f"class_to_value: ragged_offsets.{k}.offsets is not one of {OFFSET_LAYOUTS}: {layout!r}"
            )

    psl = _entry(table, "per_seq_len", "class_to_value")
    for axis in ("per_seq_len_q", "per_seq_len_kv"):
        _covers(psl, model, axis, "per_seq_len")
    per_seq_len = {}
    for k in sorted(
        set(model.values["per_seq_len_q"]) | set(model.values["per_seq_len_kv"])
    ):
        cycle = _entry(psl[k], "cycle", f"per_seq_len.{k}")
        unknown = set(psl[k]) - {"cycle", NOTE}
        if unknown:
            raise ModelError(
                f"class_to_value: per_seq_len.{k} has unknown keys {sorted(unknown)}"
            )
        if not isinstance(cycle, list) or not cycle:
            raise ModelError(
                f"class_to_value: per_seq_len.{k}.cycle is not a non-empty list"
            )
        per_seq_len[k] = [
            _length_term(x, f"per_seq_len.{k}.cycle[{i}]") for i, x in enumerate(cycle)
        ]

    paging = _entry(table, "paging", "class_to_value")
    sides_table = _entry(paging, "sides", "paging")
    paged_sides = {}
    for k in model.values["paged_kv"]:
        if k == "absent":
            continue
        letters = _entry(sides_table, k, "paging.sides")
        if (
            not isinstance(letters, list)
            or not letters
            or any(x not in ("K", "V") for x in letters)
        ):
            raise ModelError(
                f"class_to_value: paging.sides.{k} is not a non-empty list of K, V"
            )
        paged_sides[k] = list(letters)
    kv_sizes = _entry(paging, "page_size_k_vs_v", "paging")
    _covers(kv_sizes, model, "page_size_k_vs_v", "paging.page_size_k_vs_v")
    v_page_multiplier = {
        k: _integer(
            _entry(
                _entry(kv_sizes[k], "v_page_size", f"paging.page_size_k_vs_v.{k}"),
                "multiply",
                f"paging.page_size_k_vs_v.{k}.v_page_size",
            ),
            f"paging.page_size_k_vs_v.{k}.v_page_size.multiply",
        )
        for k in model.values["page_size_k_vs_v"]
    }
    sharing = _entry(paging, "page_table_sharing", "paging")
    _covers(sharing, model, "page_table_sharing", "paging.page_table_sharing")
    table_count = {"two": 2, "one": 1}
    page_tables = {}
    for k in model.values["page_table_sharing"]:
        n = _entry(sharing[k], "tables", f"paging.page_table_sharing.{k}")
        if n not in table_count:
            raise ModelError(
                f"class_to_value: paging.page_table_sharing.{k}.tables is not one or two: {n!r}"
            )
        page_tables[k] = table_count[n]
    storage = _entry(paging, "page_table_storage", "paging")
    _covers(storage, model, "page_table_storage", "paging.page_table_storage")
    packed_table = {
        k: _entry(storage[k], "ragged_offset", f"paging.page_table_storage.{k}")
        != "none"
        for k in model.values["page_table_storage"]
    }

    ahc = _entry(table, "alibi_head_count", "class_to_value")
    _covers(ahc, model, "alibi_head_count", "alibi_head_count")
    replaced = _entry(
        ahc["not_power_of_two"],
        "H_q_by_power_of_two",
        "alibi_head_count.not_power_of_two",
    )
    alibi_hq = {
        str(_integer(src, "alibi_head_count key")): _integer(
            dst, f"alibi_head_count.{src}"
        )
        for src, dst in replaced.items()
    }
    hc = _entry(
        _entry(table, "head_counts", "class_to_value"),
        "with_alibi_not_power_of_two",
        "head_counts",
    )
    alibi_heads = {}
    for hq in alibi_hq.values():
        where = f"head_counts.with_alibi_not_power_of_two[{hq}]"
        differ = _entry(
            _entry(hc, "k_and_v_differ", "head_counts.with_alibi_not_power_of_two"),
            hq,
            where,
        )
        alibi_heads[str(hq)] = {
            "mha": _integer(
                _entry(_entry(hc, "mha", where), hq, where), where + ".mha"
            ),
            "gqa": _integer(
                _entry(_entry(hc, "gqa", where), hq, where), where + ".gqa"
            ),
            "mqa": _integer(
                _entry(_entry(hc, "mqa", where), hq, where), where + ".mqa"
            ),
            "k_and_v_differ": {
                "H_k": _integer(_entry(differ, "H_k", where), where + ".H_k"),
                "H_v": _integer(_entry(differ, "H_v", where), where + ".H_v"),
            },
        }

    mma = _entry(
        _entry(table, "matmul_input_precision", "class_to_value"),
        "fp8",
        "matmul_input_precision",
    )
    mma_mode = _entry(mma, "mma_core_mode", "matmul_input_precision.fp8")
    where = "matmul_input_precision.fp8.mma_core_mode"
    mma_fp8 = {
        "when_q_is_fp8": (
            "dtype_q" if _entry(mma_mode, "when_q_is_fp8", where) == "dtype_q" else None
        ),
        "otherwise": _format(
            _entry(mma_mode, "otherwise", where), model, where + ".otherwise"
        ),
        "by_arch": {
            str(arch): {
                _format(s, model, f"{where}.by_arch"): _format(
                    d, model, f"{where}.by_arch"
                )
                for s, d in subs.items()
            }
            for arch, subs in _entry(mma_mode, "by_arch", where).items()
        },
    }
    if mma_fp8["when_q_is_fp8"] is None:
        raise ModelError(f"class_to_value: {where}.when_q_is_fp8 must be dtype_q")

    shapes = {}
    for axis, key in (
        ("bias_broadcast", "bias_shape"),
        ("dropout_mask_broadcast", "dropout_mask_broadcast"),
    ):
        t = _entry(table, key, "class_to_value")
        _covers(t, model, axis, key)
        shapes[key] = {
            k: _without_notes(_entry(t[k], "shape", f"{key}.{k}"))
            for k in model.values[axis]
        }

    bm = _entry(table, "block_mask_pattern", "class_to_value")
    _covers(bm, model, "block_mask_pattern", "block_mask_pattern")
    block_mask = {
        "layout": _without_notes(_entry(bm, "layout", "block_mask_pattern")),
        "patterns": {
            k: _without_notes(bm[k]) for k in model.values["block_mask_pattern"]
        },
    }

    sl = _entry(table, "sequence_lengths", "class_to_value")
    _covers(sl, model, "sequence_lengths", "sequence_lengths")
    sequence_lengths = {
        k: _without_notes(sl[k]) for k in model.values["sequence_lengths"]
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "model": "model/model.yaml",
            "model_sha256": model.input_files.get("model/model.yaml"),
        },
        "selection": "scale_value lists representatives, primary first; a case uses the primary",
        "batch_size": size["batch_size"],
        "q_heads": size["q_heads"],
        "seq_lengths": seq,
        "head_dim": head_dim,
        "dtype_differs": {"by_dtype_q": by_q, "by_arch": by_arch},
        "dim_order": dim_order,
        "scale_value": scale,
        "dropout_probability": dropout_p,
        "dropout_seed_offset": seed_offset,
        "dropout_mask_scale": mask_scale,
        "dropout_mask_cycle": mask_cycle,
        "page_size": page_size,
        "quantisation_factors": factors,
        "descale_lists": descale_lists,
        "ragged_tensors": ragged_tensors,
        "ragged_stats": ragged_stats,
        "ragged_offsets": ragged_offsets,
        "per_seq_len": per_seq_len,
        "paged_sides": paged_sides,
        "v_page_multiplier": v_page_multiplier,
        "page_tables": page_tables,
        "packed_page_table": packed_table,
        "alibi_head_count": {"H_q": alibi_hq, "heads": alibi_heads},
        "mma_fp8": mma_fp8,
        "bias_shape": shapes["bias_shape"],
        "dropout_mask_shape": shapes["dropout_mask_broadcast"],
        "block_mask": block_mask,
        "sequence_lengths": sequence_lengths,
    }


def render(values: dict) -> bytes:
    return (json.dumps(values, indent=2, allow_nan=False) + "\n").encode("utf-8")

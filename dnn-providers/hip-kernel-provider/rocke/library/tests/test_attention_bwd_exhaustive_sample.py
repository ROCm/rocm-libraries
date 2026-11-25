# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A covering sample of the exhaustive backward test files, for the default lane.

The modules in ``_affordable_lane.EXHAUSTIVE_MODULES`` check full cross-products
(every kernel row with every schedule pair, every IR instance, the IR golden at
every LLVM flavor) and run only in the exhaustive lane. This module calls every
CPU test function of those modules on a covering subset of its own
parameters: the parameter lists are rotated together instead of crossed, so
every value of a list of up to ``_CAP`` values appears, a longer list gives a
spread sample, and the functions measured at 0.5 s or more per case take fewer
cases (``_CAPS``: 2 cases from 0.5 s, 1 case from 2 s). A failure here is a
failure of the named exhaustive test on that case.

Left to their own modules: functions marked ``gpu`` (this module is CPU only),
and the IR golden (``test_attention_bwd_ir_golden.py``: every case at every
flavor, about 140 s at one flavor alone).
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from ._affordable_lane import EXHAUSTIVE_MODULES
from ._attention_bwd_stage_sample import _spread

_CAP = 8
# Cases per function measured at 0.5 s or more per case (prefix of the name):
# 2 from 0.5 s per case, 1 from 2 s per case.
_CAPS = {
    ("test_attention_bwd_dq", "test_dq_resource_refusals_are_exactly_the_crash"): 2,
    ("test_attention_bwd_dq", "test_kv_tile_range_exhaustive"): 2,
    ("test_attention_bwd_dq", "test_kv_tile_range_other_tiles_and_spellings"): 2,
    ("test_attention_bwd_dq_sched", "test_code_objects_keep_the_matrix"): 2,
    ("test_attention_bwd_dq_sched", "test_default_emits_no_hint"): 2,
    ("test_attention_bwd_dq_sched", "test_lds_accesses_do_not_race"): 2,
    ("test_attention_bwd_dq_sched", "test_llvm_hint_calls_equal_the_model"): 1,
    ("test_attention_bwd_dq_sched", "test_plan_and_registry_carry_dq_sched"): 2,
    ("test_attention_bwd_dq_sched", "test_scheduling_only"): 1,
    ("test_attention_bwd_ir", "test_kernarg_order_matches_params"): 1,
    ("test_attention_bwd_ir", "test_lowers_at_every_flavor"): 2,
    ("test_attention_bwd_ir", "test_new_values_change_the_ir"): 2,
    ("test_attention_bwd_ir", "test_ragged_offsets_read_once"): 2,
    ("test_attention_bwd_ir", "test_sequence_lengths_read_once"): 2,
    ("test_attention_bwd_ir", "test_validator_refuses_exactly_the_spilling"): 2,
    ("test_attention_bwd_ir", "test_worklist_reads_the_list_only_when_built"): 2,
    ("test_barrier_fence_parity_emit", "test_ll_is_the_in_process_lowering"): 2,
    ("test_attention_bwd_ring_sched", "test_code_objects_keep_the_matrix"): 2,
    ("test_attention_bwd_ring_sched", "test_dma_copy_is_followed_by_the_counted"): 2,
    ("test_attention_bwd_ring_sched", "test_llvm_hint_calls_equal_the_model"): 2,
    ("test_attention_bwd_ring_sched", "test_scheduling_only_on_the_ring"): 2,
    ("test_attention_bwd_sched_body", "test_code_objects_differ_and_keep"): 2,
    ("test_attention_bwd_sched_body", "test_default_has_no_schedule_ops"): 1,
    ("test_attention_bwd_sched_body", "test_llvm_hint_calls_equal_the_model"): 2,
    ("test_attention_bwd_sched_body", "test_read_ahead_keeps_lds_accesses"): 2,
    ("test_attention_bwd_sched_body", "test_scheduling_only"): 2,
    ("test_attention_bwd_split", "test_lds_plan_drops_the_ds_tile"): 2,
    ("test_attention_bwd_split_sched", "test_code_objects_keep_the_matrix"): 2,
    ("test_attention_bwd_split_sched", "test_dma_copy_is_followed_by_no_vmem"): 2,
    ("test_attention_bwd_split_sched", "test_every_value_is_legal_with_every"): 2,
    ("test_attention_bwd_split_sched", "test_llvm_hint_calls_equal_the_model"): 2,
    ("test_attention_bwd_split_sched", "test_scheduling_only_on_split"): 2,
    ("test_attention_bwd_unified", "test_kernarg_order_matches_params"): 2,
    ("test_attention_bwd_unified", "test_kernel_names_are_salted_and_unique"): 1,
    ("test_attention_bwd_unified", "test_lowers_at_every_flavor"): 2,
    ("test_attention_bwd_unified", "test_rebase_is_64bit"): 2,
    ("test_attention_bwd_valu", "test_batched_stats_rows_are_the_scalar_rows"): 2,
    ("test_attention_bwd_valu", "test_rne_noguard_matches_round_to_nearest_even"): 1,
    ("test_attention_bwd_xt_probe", "test_probe_kernels_use_the_intended_lds"): 2,
}
_LEFT_OUT = {"test_attention_bwd_ir_golden.py"}
_MODULES = tuple(
    importlib.import_module(f"tests.{name[:-3]}")
    for name in sorted(EXHAUSTIVE_MODULES - _LEFT_OUT)
)


def _marks(obj) -> list:
    m = getattr(obj, "pytestmark", [])
    return list(m) if isinstance(m, (list, tuple)) else [m]


def _value_lists(fn) -> list:
    """``(argnames, [(values, id, marks), ...])`` of each parametrize mark."""
    out = []
    for mark in _marks(fn):
        if mark.name != "parametrize":
            continue
        names = mark.args[0]
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",")]
        names = list(names)
        ids = mark.kwargs.get("ids")
        entries = []
        for j, v in enumerate(mark.args[1]):
            if hasattr(v, "values") and hasattr(v, "marks"):  # pytest.param
                vals, vid, vmarks = tuple(v.values), v.id, tuple(v.marks)
            else:
                vals = tuple(v) if len(names) > 1 else (v,)
                vid, vmarks = None, ()
            if vid is None and ids is not None:
                if callable(ids):
                    vid = ids(vals[0]) if len(names) == 1 else None
                else:
                    vid = ids[j]
            entries.append((vals, str(vid) if vid is not None else f"v{j}", vmarks))
        out.append((names, entries))
    return out


def _cases(mod, name, fn) -> list:
    short = mod.__name__.rsplit(".", 1)[-1]
    lists = _value_lists(fn)
    cap = next(
        (c for (m, p), c in _CAPS.items() if m == short and name.startswith(p)), _CAP
    )
    n_cases = min(max((len(e) for _, e in lists), default=1), cap)
    extra = [
        m for m in _marks(fn) + _marks(mod) if m.name in ("skipif", "xfail", "skip")
    ]
    out = []
    for i in range(n_cases):
        kwargs, ids, marks = {}, [], list(extra)
        for names, entries in lists:
            L = len(entries)
            k = i % L if L <= n_cases else _spread(list(range(L)), n_cases)[i]
            vals, vid, vmarks = entries[k]
            kwargs.update(zip(names, vals))
            ids.append(vid)
            marks += vmarks
        case_id = short.replace("test_attention_bwd_", "") + "::" + name
        if ids:
            case_id += "[" + "-".join(ids) + "]"
        out.append(pytest.param(fn, kwargs, id=case_id, marks=marks))
    return out


def _all_cases() -> list:
    out = []
    for mod in _MODULES:
        if any(m.name == "gpu" for m in _marks(mod)):
            continue
        for name, fn in vars(mod).items():
            if not name.startswith("test_") or not inspect.isfunction(fn):
                continue
            if fn.__module__ != mod.__name__ or any(
                m.name == "gpu" for m in _marks(fn)
            ):
                continue
            out += _cases(mod, name, fn)
    return out


@pytest.fixture(autouse=True)
def _bound_exhaustive_memos():
    """Clear any memo of the exhaustive modules holding more than 8 entries
    after each test, as the library conftest does for the modules it collects
    (it does not see these: their tests are not collected here)."""
    yield
    for mod in _MODULES:
        for obj in vars(mod).values():
            info = getattr(obj, "cache_info", None)
            if getattr(obj, "__module__", None) == mod.__name__ and callable(info):
                if info().currsize > 8:
                    obj.cache_clear()


@pytest.mark.parametrize("fn,kwargs", _all_cases())
def test_exhaustive_cover(fn, kwargs, request):
    """One covering case of an exhaustive test function (see the id)."""
    for name in inspect.signature(fn).parameters:
        if name not in kwargs:
            kwargs = {**kwargs, name: request.getfixturevalue(name)}
    fn(**kwargs)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the ``origami.comm`` bindings.

Every value the Python layer returns must be byte-identical to the frozen
golden CSVs the C++ test suite checks against (``tests/comm/golden/``). The
goldens are produced by the same model entry points these bindings wrap, so an
exact match proves the bindings introduce no drift: same inputs, same numbers.

  predict_row.csv        -> comm.predict_row               (us)
  collective_grid.csv    -> comm.compute_collective_latency (GPU cycles)
  tensor_collective.csv  -> comm.predict_tensor_collective  (us + shape fields)
"""

import csv
import os
import sys

import pytest

# ── Import the extension that actually carries the comm bindings ──────────
# On CI / a clean checkout, ``import origami`` resolves (via PYTHONPATH=src) to
# the extension we just built, which exposes ``origami.comm``. Some dev boxes
# have an *editable* origami install pointing at a different worktree; its
# scikit-build meta-path finder hijacks the name regardless of sys.path. We
# cannot safely load a second copy of a nanobind extension under the same
# module name in one process, so if the imported origami lacks ``comm`` and a
# redirecting finder is present, skip rather than crash the run.
def _import_origami():
    _SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import origami  # noqa: E402

    if hasattr(origami, "comm"):
        return origami

    shadow = any(
        type(f).__name__ == "ScikitBuildRedirectingFinder" for f in sys.meta_path
    )
    if shadow:
        pytest.skip(
            "an editable 'origami' from another worktree shadows this build; "
            f"it lacks comm bindings (loaded from {getattr(origami, '__file__', '?')})",
            allow_module_level=True,
        )
    raise ImportError("origami.comm is missing from the built extension")


origami = _import_origami()
comm = origami.comm

# The library no longer ships a hardcoded default system: production callers build
# one for the device they are about to run on. The goldens were frozen against the
# nominal MI300X machine, so reconstruct it explicitly here (no GPU required) from
# the calibrated gfx942 ceilings and the part's full-die topology at 2.0 GHz. This
# is byte-identical to the constants the model used to ship inline.
_GFX942 = origami.architecture_t.gfx942
_SYS = comm.make_system(
    comm.get_arch_ceilings(_GFX942),
    comm.gpu_topology_t(
        arch=_GFX942,
        num_cu=304,
        num_xcd=8,
        cu_per_xcd=38,
        l2_capacity_bytes=4 * 1024 * 1024,
    ),
    2.0,
)

# ── Golden CSV location (shared/origami/tests/comm/golden) ────────────────
_GOLDEN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tests", "comm", "golden")
)


def _golden(name):
    path = os.path.join(_GOLDEN, name)
    if not os.path.exists(path):
        pytest.skip(f"golden file not found: {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# The C++ suite uses a 1e-9 us / 1e-6 cycle absolute tolerance; the bindings
# call the identical code path, so equality should hold to full double
# precision. Keep the same conservative bounds.
_US_TOL = 1e-9
_CYCLE_TOL = 1e-6


def test_predict_row_matches_golden():
    rows = _golden("predict_row.csv")
    assert len(rows) > 300, "golden grid looks truncated"
    worst = 0.0
    for r in rows:
        got = comm.predict_row(
            r["primitive"],
            int(r["msg_bytes"]),
            int(r["world_size"]),
            int(r["num_wgs"]),
            _SYS,
        )
        exp = float(r["T_us"])
        worst = max(worst, abs(got - exp))
        assert abs(got - exp) <= _US_TOL, (
            f"predict_row {r['primitive']} W={r['world_size']} nch={r['num_wgs']} "
            f"msg={r['msg_bytes']}: got {got!r} exp {exp!r}"
        )
    print(f"predict_row: {len(rows)} rows, max |delta| = {worst:g} us")


def test_compute_collective_latency_matches_golden():
    rows = _golden("collective_grid.csv")
    assert len(rows) > 300, "golden grid looks truncated"
    worst = 0.0
    for r in rows:
        problem = comm.comm_problem_t(
            M=int(r["M"]),
            N=int(r["N"]),
            num_gpus=int(r["world_size"]),
            dtype=origami.data_type_t.BFloat16,
            split_dim=int(r["split_dim"]),
            collective=getattr(comm.primitive_t, r["primitive"]),
        )
        config = comm.comm_config_t(num_wgs=int(r["num_wgs"]))
        got = comm.compute_collective_latency(problem, config, _SYS)
        exp = float(r["T_cycles"])
        worst = max(worst, abs(got - exp))
        assert abs(got - exp) <= _CYCLE_TOL, (
            f"collective_latency {r['primitive']} W={r['world_size']} "
            f"nch={r['num_wgs']} M={r['M']} N={r['N']}: got {got!r} exp {exp!r}"
        )
    print(f"compute_collective_latency: {len(rows)} rows, max |delta| = {worst:g} cycles")


def test_predict_tensor_collective_matches_golden():
    rows = _golden("tensor_collective.csv")
    assert len(rows) > 300, "golden grid looks truncated"
    worst_us = 0.0
    for r in rows:
        shape = [int(x) for x in r["shape"].split("x")]
        p = comm.predict_tensor_collective(
            r["op"],
            shape,
            r["dtype"],
            int(r["world_size"]),
            _SYS,
            dim=int(r["dim"]),
            nchannels=int(r["nchannels"]),
            framework=r["framework"],
        )
        ctx = (
            f"{r['op']} W={r['world_size']} dim={r['dim']} nch={r['nchannels']} "
            f"fw={r['framework']} shape={r['shape']}"
        )
        for field, exp in (
            ("predicted_us", float(r["predicted_us"])),
            ("framework_overhead_us", float(r["framework_overhead_us"])),
        ):
            got = getattr(p, field)
            worst_us = max(worst_us, abs(got - exp))
            assert abs(got - exp) <= _US_TOL, f"{field} {ctx}: got {got!r} exp {exp!r}"

        assert abs(p.backend_us() - float(r["backend_us"])) <= _US_TOL, f"backend_us {ctx}"

        # Integer/shape fields must be exact.
        assert p.per_rank_bytes == int(r["per_rank_bytes"]), f"per_rank_bytes {ctx}"
        assert p.wire_bytes_per_rank == int(r["wire_bytes_per_rank"]), f"wire_bytes {ctx}"
        assert p.msg_bytes == int(r["msg_bytes"]), f"msg_bytes {ctx}"
        assert p.gpu_tile.m == int(r["gpu_tile_m"]), f"gpu_tile_m {ctx}"
        assert p.gpu_tile.n == int(r["gpu_tile_n"]), f"gpu_tile_n {ctx}"
        assert p.gpu_tile.split_dim == int(r["gpu_tile_split_dim"]), f"gpu_tile_split_dim {ctx}"
    print(f"predict_tensor_collective: {len(rows)} rows, max |delta| = {worst_us:g} us")


def test_string_and_enum_problem_agree():
    # predict_row takes the primitive by name; compute_collective_latency takes
    # the enum. For a no-shape row they must agree after the cycles->us
    # conversion the public boundary performs.
    msg_bytes, world, nch = 64 * 1024, 8, 32
    us_from_name = comm.predict_row("all_reduce", msg_bytes, world, nch, _SYS)

    # predict_row treats a bare buffer as a 1xN bf16 row (2 bytes/elem).
    problem = comm.comm_problem_t(
        M=1,
        N=msg_bytes // 2,
        num_gpus=world,
        collective=comm.primitive_t.all_reduce,
    )
    config = comm.comm_config_t(num_wgs=nch)
    cycles = comm.compute_collective_latency(problem, config, _SYS)
    us_from_enum = _SYS.gpu.cycles_to_us(cycles)

    assert abs(us_from_name - us_from_enum) <= _US_TOL

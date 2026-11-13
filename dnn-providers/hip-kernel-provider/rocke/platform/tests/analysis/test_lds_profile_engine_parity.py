# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ twin of rocke.analysis.lds must agree with the Python registry.

The C++ engine plans the attention backward LDS layouts with its own copy of
the LDS conflict predictor (``cpp/analysis/lds``, declared in
``rocke/helper_rocke.analysis.lds.h``). The Python profiles are classes, not a
data file, so the C++ profile table restates them. A new or changed Python
profile would move Python's LDS plan but not the C++ one, and the byte-identity
gate only notices when a covered config happens to pick a different swizzle.
This test closes that gap in two parts:

* ``test_cpp_profile_table_matches_registry`` reads
  ``cpp/analysis/lds/profiles.cpp`` as text (like
  ``tests/core/test_arch_table_engine_parity.py``) and compares the opcode
  table and every profile row (target, version, wave sizes, LDS capacity,
  opcodes) with the Python registry. It needs no build.
* ``test_cpp_predictor_matches_python`` runs the built ``rocke_lds_profile_dump``
  (path in ``ROCKE_LDS_PROFILE_DUMP``) and compares, for every target and
  opcode, the phase keys, the collision keys, and a set of predictions and
  refusals with Python's answers. The CTest ``rocke_lds_profile_engine_parity``
  sets the path; a plain pytest run without a build skips this part.

Source-only: it reads ``cpp/analysis/lds/profiles.cpp``, which the installed
test artifact does not carry, so ``CMakeLists.txt`` keeps it out of the
installed pytest cohort and ``tests/cmake/CoreTests.cmake`` runs it from the
source tree.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from rocke.analysis.lds import LdsAccess, predict_lds_conflicts, registered_targets
from rocke.analysis.lds.opcodes import _OPCODE_SPECS, supported_opcodes
from rocke.analysis.lds.profiles import BUILTIN_PROFILES
from rocke.analysis.lds.registry import resolve_profile
from rocke.assets import platform_root

_PROFILES_CPP = platform_root() / "cpp" / "analysis" / "lds" / "profiles.cpp"

_OPCODE_ROW = re.compile(r'\{"(\w+)", "(\w+)", (\d+), (\d+)\}')
_INT_ARRAY = re.compile(r"static const int (\w+)\[\] = \{([^}]*)\};")
_STR_ARRAY = re.compile(r"static const char\* const (\w+)\[\] = \{(.*?)\};", re.S)
_PROFILE_BLOCK = re.compile(r"\{\s*\.target = (.*?)\n    \},", re.S)
_RULES = {"ROCKE_LDS_RULES_GFX90A", "ROCKE_LDS_RULES_GFX950", "ROCKE_LDS_RULES_GFX1201"}


def _block(src: str, name: str) -> str:
    m = re.search(name + r"\[\] = \{(.*?)\n\};", src, re.S)
    assert m is not None, f"profiles.cpp has no {name} table"
    return m.group(1)


def _field(body: str, name: str) -> str:
    m = re.search(r"\." + name + r' = "?([\w]+)"?,', body)
    assert m is not None, f"profiles.cpp profile row has no .{name} field"
    return m.group(1)


def _cpp_table() -> tuple[list[tuple], list[dict]]:
    src = _PROFILES_CPP.read_text()
    opcodes = [
        (op, direction, int(width), int(align) or None)
        for op, direction, width, align in _OPCODE_ROW.findall(
            _block(src, "k_opcode_specs")
        )
    ]
    ints = {
        name: sorted(int(v) for v in body.split(","))
        for name, body in _INT_ARRAY.findall(src)
    }
    strs = {
        name: sorted(re.findall(r'"(\w+)"', body))
        for name, body in _STR_ARRAY.findall(src)
    }
    profiles = []
    for m in _PROFILE_BLOCK.finditer(_block(src, "k_profiles")):
        body = ".target = " + m.group(1) + ","
        profiles.append(
            {
                "target": _field(body, "target"),
                "profile_version": int(_field(body, "profile_version")),
                "supported_wave_sizes": ints[_field(body, "supported_wave_sizes")],
                "lds_capacity_bytes": int(_field(body, "lds_capacity_bytes")),
                "supported_opcodes": strs[_field(body, "supported_opcodes")],
                "rules": _field(body, "rules"),
            }
        )
    return opcodes, profiles


def _python_profile(p) -> dict:
    return {
        "target": p.identity.target,
        "profile_version": p.identity.profile_version,
        "supported_wave_sizes": sorted(p.supported_wave_sizes),
        "lds_capacity_bytes": p.lds_capacity_bytes,
        "supported_opcodes": sorted(p.supported_opcodes),
    }


def test_cpp_profile_table_matches_registry():
    opcodes, profiles = _cpp_table()
    assert opcodes == [
        (s.opcode, s.direction, s.access_width_bytes, s.alignment_bytes)
        for s in _OPCODE_SPECS.values()
    ], "cpp/analysis/lds/profiles.cpp k_opcode_specs differs from opcodes._OPCODE_SPECS"
    for row in profiles:
        assert row.pop("rules") in _RULES, row
    assert profiles == [_python_profile(p) for p in BUILTIN_PROFILES], (
        "cpp/analysis/lds/profiles.cpp k_profiles differs from "
        "rocke.analysis.lds.profiles.BUILTIN_PROFILES; update the C++ twin"
    )


# ---------------------------------------------------------------- behavior


def _ints(values) -> str:
    return ",".join(str(v) for v in values)


def _py_table() -> list[str]:
    out = [
        f"opcode {s.opcode} {s.direction} {s.access_width_bytes} {s.alignment_bytes}"
        for s in _OPCODE_SPECS.values()
    ]
    out.append("targets " + " ".join(registered_targets()))
    for p in BUILTIN_PROFILES:
        out.append(
            f"profile {p.identity.target} version={p.identity.profile_version} "
            f"wave_sizes={_ints(sorted(p.supported_wave_sizes))} "
            f"capacity={p.lds_capacity_bytes} "
            f"opcodes={','.join(sorted(p.supported_opcodes))}"
        )
    return out


def _py_keys(target: str, opcode: str) -> list[str]:
    try:
        p = resolve_profile(target)
    except ValueError:
        return [f"keys {target} {opcode}: no profile"]
    try:
        phases = " ".join(_ints(p.phase_key(opcode, lane)) for lane in range(64))
        classes = _ints(p.collision_key(opcode, a) for a in range(1024))
    except ValueError:
        return [f"keys {target} {opcode}: unsupported"]
    return [f"phase {target} {opcode} {phases}", f"class {target} {opcode} {classes}"]


def _py_predict(
    target: str, opcode: str, wave: int, acc: list[tuple]
) -> tuple[str, str]:
    label = f"{target} {opcode} {wave}" + "".join(
        f" {i}:{lane}:{addr}:{w}:{int(act)}" for i, lane, addr, w, act in acc
    )
    try:
        recs = tuple(
            LdsAccess(
                access_id=i,
                lane=lane,
                lds_byte_address=addr,
                access_width_bytes=w,
                active=act,
            )
            for i, lane, addr, w, act in acc
        )
        s = predict_lds_conflicts(
            target=target, opcode=opcode, wave_size=wave, accesses=recs
        )
    except ValueError as exc:
        return label, f"predict {label}: ValueError: {exc}"
    s = s.summary
    return label, (
        f"predict {label}: {s.active_access_count} {s.conflicted_access_count} "
        f"{s.broadcast_access_count} {s.inactive_access_count} "
        f"{s.conflict_group_count} {s.maximum_multiplicity}"
    )


def _requests(cpp_targets: list[str]) -> tuple[list[str], list[str]]:
    """The commands for the dump program and the Python answers, in order."""
    cmds, want = ["table"], _py_table()
    targets = sorted(set(registered_targets()) | set(cpp_targets))
    for target in targets:
        for opcode in supported_opcodes():
            cmds.append(f"keys {target} {opcode}")
            want += _py_keys(target, opcode)
    for target in targets + ["gfx000"]:
        for opcode in (*supported_opcodes(), "DS_READ_B32", "ds_bogus"):
            spec = _OPCODE_SPECS.get(opcode.lower())
            w = spec.access_width_bytes if spec else 4
            cases = []
            for wave in (32, 64):
                for stride in (0, 4, 8, 16, 64, 128, 256, 260):
                    cases.append(
                        (wave, [(i, i, i * stride, w, True) for i in range(wave)])
                    )
            for gap in (0, 2, 4, 8, 16, 128, 256):
                cases.append((64, [(0, 0, 0, w, True), (1, 1, gap, w, True)]))
                cases.append((32, [(0, 0, 0, w, True), (1, 1, gap, w, True)]))
            cases += [
                (64, [(0, 0, 0, w, True), (1, 32, 0, w, False), (2, 33, 128, w, True)]),
                (64, [(1, 0, 0, w, True), (0, 1, 128, w, True)]),
                (64, [(0, 0, 0, w, True), (0, 1, 128, w, True)]),
                (64, [(0, 0, 0, w, True), (1, 0, 128, w, True)]),
                (32, [(0, 31, 0, w, True), (1, 32, 128, w, True)]),
                (64, [(0, 0, 0, w + 4, True)]),
                (64, [(0, 0, 65536 - w, w, True), (1, 1, 65536, w, True)]),
                (64, [(0, 0, 163840, w, True)]),
                (64, [(0, -1, 0, w, True)]),
                (64, [(-1, 0, 0, w, True)]),
                (64, [(0, 0, -4, w, True)]),
                (64, [(0, 0, 0, 0, True)]),
                (16, [(0, 0, 0, w, True)]),
                (64, []),
            ]
            for wave, acc in cases:
                label, line = _py_predict(target, opcode, wave, acc)
                cmds.append(f"predict {label}")
                want.append(line)
    return cmds, want


def test_cpp_predictor_matches_python():
    dump = os.environ.get("ROCKE_LDS_PROFILE_DUMP")
    if not dump:
        pytest.skip(
            "ROCKE_LDS_PROFILE_DUMP is not set; the CTest "
            "rocke_lds_profile_engine_parity runs this part against the build"
        )
    _, profiles = _cpp_table()
    cmds, want = _requests([row["target"] for row in profiles])
    got = subprocess.run(
        [dump], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=True
    ).stdout.splitlines()
    diff = [(w, g) for w, g in zip(want, got) if w != g]
    assert len(got) == len(want), f"{len(got)} answers for {len(want)} expected lines"
    assert (
        not diff
    ), "C++ LDS predictor differs from rocke.analysis.lds; first:\n" + "\n".join(
        f"python: {w}\nc++:    {g}" for w, g in diff[:3]
    )

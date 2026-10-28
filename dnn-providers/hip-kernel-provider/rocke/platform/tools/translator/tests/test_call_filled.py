# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A call of a function (or method) that did not translate but is filled by
hand (holes/<module>/<f>.cpp) goes through the fill's C entry: the header
declares it and the fill defines it, so its callers translate instead of
being refused whole ("call to X, which is not translated"). Without the fill
the caller stays a hole."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import ROCKE, TR, platform_probe, strip_holes

# probe_valid is refused (validate_io has no rule) and filled; probe_build
# calls it and the filled method BatchedTranspose2DSpec.kernel_name
PROBE = """\
def probe_valid(spec: BatchedTranspose2DSpec, arch: str = "gfx950") -> tuple[bool, str]:
    ok, why = validate_io(
        IOSpecRule(dtype=spec.dtype, block_size=spec.block_size, vec=spec.vec)
    )
    if not ok:
        return ok, why
    return True, "ok"


def probe_build(spec: BatchedTranspose2DSpec, arch: str = "gfx950") -> int:
    ok, why = probe_valid(spec, arch)
    if not ok:
        raise ValueError(f"invalid probe spec: {why}")
    name = spec.kernel_name()
    return len(name) + spec.tile_m
"""

FILLS = {
    "probe_valid": """\
/* probe_valid: validate_io(IOSpecRule(...)), then (True, "ok") */
bool rocke_batched_transpose_probe_valid(const rocke_batched_transpose_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    (void)arch;
    rocke_io_spec_rule_t rule;
    const char* why = NULL;
    const int bs = spec->vec != 0 ? (spec->tile_m * spec->tile_n) / spec->vec : 0;
    rocke_io_spec_rule_init(&rule, spec->dtype, bs, spec->vec);
    const bool ok = rocke_validate_io(NULL, &rule, &why);
    rocke_spec_set_reason(reason, reason_cap, ok ? "ok" : why);
    return ok;
}
""",
    "BatchedTranspose2DSpec_kernel_name": """\
/* kernel_name: kernel_name_join(name, dtype, f"{tile_m}x{tile_n}", f"v{vec}", f"p{lds_pad}") */
rocke_status_t rocke_batched_transpose_spec_kernel_name(const rocke_batched_transpose_spec_t* self,
                                                        char* out,
                                                        size_t out_cap)
{
    const std::string t = std::to_string(self->tile_m) + "x" + std::to_string(self->tile_n);
    const std::string v = "v" + std::to_string(self->vec);
    const std::string p = "p" + std::to_string(self->lds_pad);
    const char* parts[4] = {self->dtype, t.c_str(), v.c_str(), p.c_str()};
    return rocke_kernel_name_join(self->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}
""",
}


def _run(env, abi, out=None):
    """tr.abi, then tr.translate (--out when given, else --analyze)."""
    run = None
    for args in (
        ["tr.abi", "batched_transpose", "--header", str(abi)],
        ["tr.translate", "batched_transpose", "--abi", str(abi)]
        + (["--out", str(out)] if out else ["--analyze"]),
    ):
        run = subprocess.run(
            [sys.executable, "-m", *args, "--rocke", str(ROCKE)],
            cwd=TR,
            env=env,
            capture_output=True,
            text=True,
        )
    return run.stdout + run.stderr


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """(analyze output, source, abi dir) with the fills."""
    root = tmp_path_factory.mktemp("call_filled")
    env = platform_probe(root, PROBE, FILLS)
    abi = root / "abi"
    abi.mkdir()
    out = root / "batched_transpose.cpp"
    _run(env, abi, out)
    return _run(env, abi), out.read_text(), abi


def test_caller_of_fills_translates(probe):
    analyze = probe[0]
    assert re.search(r"^filled +probe_valid ", analyze, re.M), analyze
    assert re.search(r"^filled +BatchedTranspose2DSpec.kernel_name ", analyze, re.M)
    assert re.search(r"^translated +probe_build$", analyze, re.M), analyze


def test_caller_calls_the_fill_entries_in_order(probe):
    """The Python order (validate, raise with its text, then kernel_name),
    each through the fill's C entry and its header ABI."""
    cpp = re.sub(r"\s+", " ", probe[1])
    m = re.search(r"int64_t py_probe_build\(.*?return \(py::len\(name\)", cpp)
    assert m, probe[1]
    body = m.group(0)
    i_valid = body.index("rocke_batched_transpose_probe_valid(spec, arch, reason_")
    i_raise = body.index(
        'throw ckc::ValueError(std::string("invalid probe spec: ") + py::str(why));'
    )
    i_name = body.index("rocke_batched_transpose_spec_kernel_name(spec, buf_")
    assert i_valid < i_raise < i_name, body
    # (ok, reason) is the Python tuple, not a raise
    assert "std::tuple<bool, std::string>{ok_" in body, body
    # a str result that does not fit raises
    assert '!= ROCKE_OK) throw ckc::ValueError("string result does not fit");' in body


def test_source_compiles(probe, tmp_path):
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    _, cpp, abi = probe
    inc = tmp_path / "inc"
    shutil.copytree(abi, inc / "rocke")
    src = tmp_path / "src.cpp"
    src.write_text(strip_holes(cpp))
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-Wall", "-Werror", "-I", str(inc)]
    cmd += ["-isystem", str(ROCKE / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_without_fill_the_caller_is_a_hole(tmp_path):
    env = platform_probe(tmp_path, PROBE)
    abi = tmp_path / "abi"
    abi.mkdir()
    analyze = _run(env, abi)
    assert re.search(
        r"^hole +probe_build \(call to probe_valid, which is not translated",
        analyze,
        re.M,
    ), analyze

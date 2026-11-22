# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One library, two solutions sharing a handwritten custom kernel.

A handwritten CustomKernel is named by CustomKernel.name alone, so a beta-on
solution and its bespoke (BespokeNarrowing) UseBeta:false twin share one kernel
(correctly: the assembly is the same file). Their helper kernels are not
shared: the bespoke UseBeta:false twin selects the `_Beta0` conversion
helpers. TensileCreateLibrary must build the helpers of every solution it
writes into the library, not only those of the solution that represents the
shared kernel.

Fixture gfx950_TN_wvSpltK_m1.yaml: stock solution 0 (wvSpltK_bf16_tn_m1) of
asm_full/gfx950/gfx950/Range/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml
(sha256 519827cc64061853aed91246aff39229674db10802836bd2aa2147f1a65bc37f),
extracted unchanged.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import struct
import sys

import pytest

from conversion_helper_common import DATA, beta_variant
from bespoke_codegen_common import (
    DATA as NARROW_DATA, compile_helpers, disassemble, variant, workspace_waterfalls,
)
from measure_generation import _elfSections

pytestmark = pytest.mark.unit
RECIPE = "gfx950_TN_wvSpltK_m1.yaml"


def built_helpers(tmp_path, label, logic_files, loading=()):
    """Run TensileCreateLibrary on logic_files; return the helper kernels it built."""
    logic = tmp_path / label / "logic"
    out = tmp_path / label / "out"
    logic.mkdir(parents=True)
    for i, path in enumerate(logic_files):
        shutil.copy(path, logic / f"{i:03d}_{path.name}")
    cxx = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path),
               TENSILE_HELPER_CACHE_DIR=str(tmp_path / label / "helper_cache"))
    res = subprocess.run(
        [sys.executable, "-m", "tensilelite.TensileCreateLibrary", "--architecture=gfx950",
         f"--cxx-compiler={cxx}", "--keep-build-tmp", "--no-compress", "--jobs=1", *loading,
         str(logic), str(out), "HIP"],
        capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    (kernels_cpp,) = out.glob("build_tmp/*/Kernels.cpp")
    return set(re.findall(r"__global__\s+void\s+(\w+)\s*\(", kernels_cpp.read_text()))


# The combined build runs with lazy loading (the default; solutions sit in the
# lazy libraries) and with --no-lazy-library-loading (solutions sit in the
# master library). The single builds are the default-mode reference.
@pytest.mark.parametrize("loading", [(), ("--no-lazy-library-loading",)], ids=["lazy", "no-lazy"])
def test_shared_handwritten_kernel_keeps_each_solutions_helpers(tmp_path, loading):
    general = DATA / RECIPE
    (tmp_path / "variants").mkdir()
    beta_off = beta_variant(general, tmp_path / "variants" / "beta_off.yaml", False, bespoke=True)

    alone_general = built_helpers(tmp_path, "general", [general])
    alone_beta_off = built_helpers(tmp_path, "beta_off", [beta_off])
    assert alone_beta_off and alone_beta_off.isdisjoint(alone_general), \
        "fixture no longer exercises distinct beta-on/off helpers"

    together = built_helpers(tmp_path, "together", [general, beta_off], loading)
    assert not (alone_general - together), \
        f"beta-on helpers missing: {sorted(alone_general - together)}"
    assert not (alone_beta_off - together), \
        f"UseBeta:false helpers missing: {sorted(alone_beta_off - together)}"


def conversion_text_digests(path):
    """Exact per-function .text bytes, using ELF symbol sizes (not alignment padding)."""
    data = path.read_bytes()
    sections = _elfSections(data)
    _, symoff, symsize = sections[".symtab"]
    _, stroff, _ = sections[".strtab"]
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, = struct.unpack_from("<H", data, 0x3A)
    digests = {}
    for offset in range(symoff, symoff + symsize, 24):
        nameoff, info, _, section, value, size = struct.unpack_from("<IBBHQQ", data, offset)
        name = data[stroff + nameoff:data.index(b"\0", stroff + nameoff)].decode()
        if info & 15 != 2 or not name.startswith("Cijk_") or "_PostGSU" not in name:
            continue
        _, _, _, address, fileoff, section_size = struct.unpack_from(
            "<IIQQQQ", data, shoff + section * shentsize)
        start = value - address
        assert size > 0 and 0 <= start < start + size <= section_size, name
        digests[name] = hashlib.sha256(data[fileoff + start:fileoff + start + size]).hexdigest()
    assert digests, "code object has no sized conversion function symbols"
    return digests


@pytest.mark.parametrize("bespoke_first", [False, True], ids=["stock-first", "bespoke-first"])
@pytest.mark.parametrize("loading", [(), ("--no-lazy-library-loading",)], ids=["lazy", "no-lazy"])
def test_mixed_beta_kept_library_preserves_stock_and_isolates_bespoke_helpers(tmp_path, bespoke_first, loading):
    # Feature: give beta-kept bespoke helpers their own name before name-based
    # dedup. Keeping the shared name either loses the bespoke body (stock first)
    # or replaces stock machine code (bespoke first). Stock pins catch both a
    # shared uniformity change and an incorrectly deduplicated bespoke body.
    problem = {"UseBeta": True, "ActivationType": "none", "UseBias": 0,
               "BiasDataTypeList": [], "UseScaleAlphaVec": 0}
    stock = variant("bf16_streamk.yaml", tmp_path / "stock.yaml", problem=problem,
                    solution={"BespokeNarrowing": False})
    bespoke = variant("bf16_streamk.yaml", tmp_path / "bespoke.yaml", problem=problem,
                      solution={"BespokeNarrowing": True})
    inputs = [bespoke, stock] if bespoke_first else [stock, bespoke]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
        built_helpers(tmp_path, "mixed", inputs, loading)
    (obj,) = (tmp_path / "mixed/out/library/gfx950").glob("Kernels.so-*.hsaco")
    code = disassemble(obj)
    actual = conversion_text_digests(obj)
    pins = json.loads((NARROW_DATA / "stock_plain_conversion_digests.json").read_text())
    assert set(pins) <= set(actual), "mixed library lost stock conversion helpers"
    assert {name: actual[name] for name in pins} == pins, \
        "mixed library replaced stock helper machine-code bytes"
    expected = {f"Cijk_SB{'_GG' if grouped else ''}_Bespoke_PostGSU{gsu}_VW{vw}"
                for grouped in (False, True) for gsu in (2, 4, 8, 16) for vw in (1, 2, 4)}
    assert set(actual) == set(pins) | expected, \
        "mixed library did not retain separately named stock and bespoke conversion bodies"
    for name in sorted(expected):
        gsu = int(re.search(r"_PostGSU(\d+)_", name)[1])
        loads = {2: 2, 4: 4, 8: 8, 16: 152}[gsu]
        assert sum(text.startswith("buffer_load_") for _, text in code[name]) == loads, \
            f"{name}: mixed library lost W reduction loads"
        assert not workspace_waterfalls(code[name]), f"{name}: mixed bespoke helper waterfalls W loads"


def test_plain_stock_conversion_bodies_are_byte_identical(tmp_path):
    # Preservation: a shared descriptor change must not alter stock helper code.
    # Adding a lane-dependent record limit in memory_gfx.h breaks these byte pins.
    stock = variant("bf16_streamk.yaml", tmp_path / "stock.yaml",
                    problem={"UseBeta": True, "ActivationType": "none", "UseBias": 0,
                             "BiasDataTypeList": [], "UseScaleAlphaVec": 0},
                    solution={"BespokeNarrowing": False})
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
        compile_helpers(stock, tmp_path / "compiled")
    (obj,) = (tmp_path / "compiled/library/gfx950").glob("Kernels.so-*.hsaco")
    pins = json.loads((NARROW_DATA / "stock_plain_conversion_digests.json").read_text())
    assert conversion_text_digests(obj) == pins, "stock plain conversion helper machine-code bytes changed"

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# StinkyTofu IR checkpoint round trip through a TensileLite build: a build with
# --dump-st-ir writes one bundle per kernel, and a build with --load-st-ir on
# the unedited bundles must emit byte-identical assembly to a plain build.
################################################################################

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensilelite.Common.GlobalParameters import DEFAULT_ST_IR_CHECKPOINT
from tensilelite.KernelWriter import stinkyTofuIRCheckpointOptions

pytestmark = pytest.mark.unit

_CONFIG = (Path(__file__).parent / "test_data" / "st_ir_checkpoint_gfx1250.yaml").resolve()


def _gfx1250_asm_supported() -> bool:
    import shutil, tempfile
    asm = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not os.path.exists(asm):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.s")
        Path(src).write_text("v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[8:23], v[8:23], v[0:7]\n")
        r = subprocess.run(
            [asm, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1250", "-c", src, "-o", os.path.join(d, "t.o")],
            capture_output=True,
        )
        return r.returncode == 0


def _build(output_dir: Path, tensile_args: list[str], *extra: str) -> str:
    """One --build-only run in a fresh interpreter, so global parameters cannot leak."""
    args = [str(_CONFIG), str(output_dir), "--build-only", *tensile_args,
            "--gpu-targets", "gfx1250", *extra]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; from tensilelite.run import main; main(sys.argv[1:])", *args],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r.stdout + r.stderr


def _assembly(output_dir: Path) -> dict[str, bytes]:
    files = {p.name: p.read_bytes() for p in output_dir.rglob("*.s")}
    assert files, f"no assembly under {output_dir}"
    return files


def test_checkpoint_options_default_spec(tmp_path: Path) -> None:
    assert stinkyTofuIRCheckpointOptions({}) == {
        "IRCheckpointAfterPass": "", "IRCheckpointDumpDir": "", "IRCheckpointLoadDir": ""}
    opts = stinkyTofuIRCheckpointOptions({"StinkyTofuIRDumpPath": str(tmp_path)})
    assert opts["IRCheckpointAfterPass"] == DEFAULT_ST_IR_CHECKPOINT
    assert opts["IRCheckpointDumpDir"] == str(tmp_path)
    opts = stinkyTofuIRCheckpointOptions({"StinkyTofuIRLoadPath": "rel",
                                          "StinkyTofuIRCheckpointAfterPass": "module:X#2"})
    assert opts["IRCheckpointAfterPass"] == "module:X#2"
    assert os.path.isabs(opts["IRCheckpointLoadDir"])


@pytest.mark.skipif(not _gfx1250_asm_supported(),
                    reason="assembler does not support gfx1250 WMMA")
def test_st_ir_checkpoint_reload_is_byte_identical(tensile_args: list[str],
                                                   tmp_path: Path) -> None:
    bundles = tmp_path / "bundles"
    plain = _assembly_of(tmp_path / "plain", tensile_args)

    log = _build(tmp_path / "dump", tensile_args, "--dump-st-ir", str(bundles))
    assert "IRCheckpoint: dumped" in log
    assert _assembly(tmp_path / "dump") == plain

    metas = list(bundles.glob("*/meta.json"))
    assert len(metas) == len(plain)
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        assert meta["spec"] == DEFAULT_ST_IR_CHECKPOINT + "#1"
        assert meta["arch"] == [12, 5, 0]
        stir = (meta_path.parent / meta["stir"]).read_text()
        assert "mod.ckpt = { id = 0 }" in stir
        assert stir.count("mod.ckpt") == meta["instructions"]

    log = _build(tmp_path / "load", tensile_args, "--load-st-ir", str(bundles))
    assert log.count("IRCheckpoint: loaded") == len(plain)
    assert "(0 moved, 0 new or changed, 0 removed)" in log
    assert _assembly(tmp_path / "load") == plain


def _assembly_of(output_dir: Path, tensile_args: list[str]) -> dict[str, bytes]:
    _build(output_dir, tensile_args)
    return _assembly(output_dir)

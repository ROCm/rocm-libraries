# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import convert_mlrec_v1_to_v2 as cli
from lib import mlrec
from lib.hardware import arch_constants

from test_mlrec import LABELS, random_bundle, v1_from_v2


def v2_bytes(arch="gfx1250v0"):
    return mlrec.write_model(
        random_bundle(LABELS, seed=8), None, arch, arch_constants(arch), "int8"
    )


def test_output_file(tmp_path):
    v2 = v2_bytes()
    src = tmp_path / "m.bin"
    src.write_bytes(v1_from_v2(v2))
    assert cli.main([str(src), "-o", str(tmp_path / "out.bin")]) == 0
    assert (tmp_path / "out.bin").read_bytes() == v2
    assert src.read_bytes()[:8] == b"MLREC_v1"


def test_in_place_and_out_dir(tmp_path, capsys):
    v2 = v2_bytes("gfx950")
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(v1_from_v2(v2, trailer=False))
    b.write_bytes(v2)
    assert cli.main(["--out-dir", str(tmp_path / "o"), str(a), str(b)]) == 0
    assert (tmp_path / "o" / "a.bin").read_bytes() == v2
    assert (tmp_path / "o" / "b.bin").read_bytes() == v2
    assert "already MLREC_v2" in capsys.readouterr().out
    assert cli.main(["--in-place", str(a)]) == 0
    assert a.read_bytes() == v2


def test_explicit_arch_and_errors(tmp_path):
    v2 = v2_bytes("gfx950")
    src = tmp_path / "m.bin"
    src.write_bytes(v1_from_v2(v2))
    assert cli.main([str(src), "-o", str(tmp_path / "x.bin"), "--arch", "gfx1250"]) == 0
    parsed = mlrec.read_model((tmp_path / "x.bin").read_bytes())
    assert parsed["arch"] == "gfx950" and parsed["constants"] == arch_constants(
        "gfx1250"
    )
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a model")
    assert cli.main([str(bad), "--in-place"]) == 1
    assert bad.read_bytes() == b"not a model"

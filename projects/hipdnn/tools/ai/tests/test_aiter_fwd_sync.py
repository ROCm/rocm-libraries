# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-aiter-asm-sdpa-fwd-sync skill's aiter_fwd_sync.py.

Each test builds a small AITER-shaped git repository (two tagged commits) and a
rocm-libraries-shaped tree in tmp_path, then drives the script's CLI. Objects are
minimal ELF files carrying only a symbol table, which is all the script reads.
No test touches the network or the real checkout.
"""

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_AI_DIR = Path(__file__).resolve().parent.parent
SCRIPT = (
    TOOLS_AI_DIR
    / "skills"
    / "hipdnn-aiter-asm-sdpa-fwd-sync"
    / "scripts"
    / "aiter_fwd_sync.py"
)

HEADER = "dtype,hdim_q,hdim_v,mask,mode,bf16_cvt,ts_qo,ts_kv,knl_name,co_name"
ROW_HD128 = "bf16,128,128,0,0,0,256,32,_ZN5aiter8fwd_hd128E,fwd_hd128.co"
ROW_HD128_RTNA = (
    "bf16,128,128,0,0,1,256,32,_ZN5aiter13fwd_hd128_rtnaE,fwd_hd128_rtna.co"
)
ROW_FP8 = "fp8bf16,128,128,0,0,1,256,64,_ZN5aiter6fwd_fp8E,fwd_fp8.co"
ROW_HD256 = "bf16,256,256,0,0,0,128,32,_ZN5aiter8fwd_hd256E,fwd_hd256.co"
ROW_HD128_GROUP = "bf16,128,128,0,1,0,256,32,_ZN5aiter10fwd_hd128gE,fwd_hd128g.co"
ROW_FP8_GROUP = "fp8bf16,128,128,0,1,1,256,64,_ZN5aiter8fwd_fp8gE,fwd_fp8g.co"

HIP_COMMON_H = """\
struct p3
{
    unsigned int _p0, _p1, _p2;
};
struct p2
{
    unsigned int _p0, _p1;
};
struct p1
{
    unsigned int _p0;
};
"""

MHA_FWD_H = """\
#pragma once
struct __attribute__((packed)) fmha_fwd_v3_args
{
    void* ptr_o;
    p2 _p0;
};
"""

MHA_FWD_CU = """\
#include "mha_fwd.h"
float fmha_fwd_v3(mha_fwd_args a, const stream_config& s);
int get_cfg_mask_type(const mha_fwd_args& a)
{
    return 0;
}
std::string get_kernel_name_key(const std::string& arch_id)
{
    return arch_id;
}
std::string get_kernel_co_name(const std::string& cfg_co_name, const std::string& arch_id)
{
    return cfg_co_name;
}
void init_fmha_fwd_v3_args(fmha_fwd_v3_args& args, const mha_fwd_args& a)
{
    args.ptr_o = a.o_ptr;
}
std::tuple<int, int, int> get_grid_dim(const mha_fwd_args& a, int ts_qo)
{
    return {1, 1, 1};
}
float fmha_fwd_v3(mha_fwd_args a, const stream_config& s)
{
    if(a.hdim_q != 128)
        return -1;
    return 0;
}
float fmha_fwd_ck(mha_fwd_args a, const stream_config& s)
{
    return 0;
}
"""


def elf(*symbols: str) -> bytes:
    """A 64-bit little-endian ELF holding a symbol table with the given names."""
    strtab = b"\0" + b"".join(s.encode() + b"\0" for s in symbols)
    offsets, at = [], 1
    for s in symbols:
        offsets.append(at)
        at += len(s) + 1
    symtab = b"\0" * 24 + b"".join(
        struct.pack("<IBBHQQ", o, 0x12, 0, 1, 0, 0) for o in offsets
    )
    shstrtab = b"\0.strtab\0.symtab\0.shstrtab\0"
    body_at = 64
    str_at = body_at
    sym_at = str_at + len(strtab)
    shs_at = sym_at + len(symtab)
    sh_at = shs_at + len(shstrtab)
    header = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    header += struct.pack(
        "<HHIQQQIHHHHHH", 1, 224, 1, 0, 0, sh_at, 0, 64, 0, 0, 64, 4, 3
    )
    sections = [
        struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, str_at, len(strtab), 0, 0, 1, 0),
        struct.pack("<IIQQQQIIQQ", 9, 2, 0, 0, sym_at, len(symtab), 1, 1, 8, 24),
        struct.pack("<IIQQQQIIQQ", 17, 3, 0, 0, shs_at, len(shstrtab), 0, 0, 1, 0),
    ]
    return header + strtab + symtab + shstrtab + b"".join(sections)


def kernel(row: str, salt: str = "") -> bytes:
    knl = row.split(",")[-2]
    return elf(knl, f"{knl}.kd") + salt.encode()


def csv_text(*rows: str, crlf: bool = False) -> str:
    newline = "\r\n" if crlf else "\n"
    return newline.join([HEADER, *rows]) + newline


@pytest.fixture(scope="module")
def sync() -> ModuleType:
    spec = importlib.util.spec_from_file_location("aiter_fwd_sync", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class Aiter:
    """An AITER-shaped repository whose commits the tests write."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir()
        git(root, "init", "-q", "-b", "main")

    def write(self, path: str, data: bytes | str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode())

    def remove(self, path: str) -> None:
        (self.root / path).unlink()

    def module(
        self, arch: str, rows: list[str], roots=("",), crlf=False, ck=()
    ) -> None:
        base = f"hsa/{arch}/fmha_v3_fwd"
        self.write(f"{base}/fmha_fwd.csv", csv_text(*rows, crlf=crlf))
        for root in roots:
            for row in rows:
                co = row.split(",")[-1]
                knl = row.split(",")[-2]
                data = elf(knl, "_ZN7ck_tile6kentryE") if knl in ck else kernel(row)
                self.write(f"{base}/{root}/{co}".replace("//", "/"), data)

    def commit(self, tag: str) -> str:
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", tag)
        git(self.root, "tag", tag)
        return git(self.root, "rev-parse", "HEAD").strip()


def make_tree(root: Path, archs=("gfx942",)) -> Path:
    engine = root / "dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine"
    kernels = engine / "asm/asm_kernels"
    kernels.mkdir(parents=True)
    (engine / "CMakeLists.txt").write_text(
        f"set(ASM_SDPA_GPU_ARCHS {' '.join(archs)})\n-m fmha_v3_fwd\n-m fmha_v3_bwd\n"
    )
    return kernels


def vendor(kernels: Path, arch: str, rows: list[str], roots=("",)) -> None:
    base = kernels / arch / "fmha_v3_fwd"
    base.mkdir(parents=True)
    (base / "fmha_fwd.csv").write_text(csv_text(*rows))
    for root in roots:
        for row in rows:
            path = base / root / row.split(",")[-1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(kernel(row))


@pytest.fixture
def env(tmp_path):
    aiter = Aiter(tmp_path / "aiter")
    tree = tmp_path / "rocm-libraries"
    kernels = make_tree(tree)
    cache = tmp_path / "cache"
    report = tmp_path / "out" / "plan.json"
    return aiter, tree, kernels, cache, report


def run(sync, tree: Path, aiter: Aiter, cache: Path, *args: str) -> int:
    return sync.main(
        [
            "--repo-root",
            str(tree),
            "--upstream",
            str(aiter.root),
            "--cache",
            str(cache),
            *args,
        ]
    )


def base_upstream(aiter: Aiter) -> str:
    aiter.write("csrc/include/mha_fwd.h", MHA_FWD_H)
    aiter.write("csrc/include/aiter_hip_common.h", HIP_COMMON_H)
    aiter.write("csrc/cpp_itfs/mha_fwd.cu", MHA_FWD_CU)
    aiter.write("hsa/codegen.py", "# codegen\n")
    aiter.module(
        "gfx942", [ROW_HD128, ROW_HD128_RTNA, ROW_FP8], roots=("MI300", "MI308")
    )
    return aiter.commit("v0.1.0")


def findings(report: Path, key: str) -> list[dict]:
    return json.loads(report.read_text())["modules"][key]["findings"]


def kinds(report: Path, key: str) -> set[str]:
    return {f["kind"] for f in findings(report, key)}


def test_round_trip_keeps_the_vendored_subset_and_verifies(sync, env, capsys):
    """An unchanged release re-pins the tree: rows absent at the baseline stay out,
    CRLF upstream CSVs compare equal, and verify passes after apply."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.module(
        "gfx942",
        [ROW_HD128, ROW_HD128_RTNA, ROW_FP8],
        roots=("MI300", "MI308"),
        crlf=True,
    )
    aiter.commit("v0.2.0")

    assert (
        run(sync, tree, aiter, cache, "verify") == 1
    )  # unpinned before the first update
    assert (
        run(
            sync,
            tree,
            aiter,
            cache,
            "plan",
            "--ref",
            "v0.2.0",
            "--arch",
            "gfx942",
            "--baseline",
            base,
            "--report",
            str(report),
        )
        == 0
    )
    plan = json.loads(report.read_text())["modules"]["gfx942/fmha_v3_fwd"]
    assert plan["tier"] == 0, plan["findings"]
    assert [r["knl_name"] for r in plan["excluded_rows"]] == [
        "_ZN5aiter13fwd_hd128_rtnaE"
    ]

    assert run(sync, tree, aiter, cache, "apply", "--report", str(report)) == 0
    assert run(sync, tree, aiter, cache, "verify") == 0
    module = kernels / "gfx942/fmha_v3_fwd"
    assert not (module / "MI300/fwd_hd128_rtna.co").exists()
    assert (module / "fmha_fwd.csv").read_text() == csv_text(ROW_HD128, ROW_FP8)
    pins = json.loads((kernels / "aiter_fwd_pins.json").read_text())
    assert pins["modules"]["gfx942/fmha_v3_fwd"]["tag"] == "v0.2.0"
    assert "SHA256 manifest (5 files)" in (module / "SOURCE.md").read_text()


def test_new_rows_and_changed_objects_are_tiered(sync, env):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.module(
        "gfx942",
        [ROW_HD128, ROW_HD128_RTNA, ROW_FP8, ROW_HD256],
        roots=("MI300", "MI308"),
    )
    aiter.write(
        "hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128.co", kernel(ROW_HD128, "rebuilt")
    )
    aiter.commit("v0.2.0")

    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.2.0",
        "--arch",
        "gfx942",
        "--baseline",
        base,
        "--report",
        str(report),
    )
    items = findings(report, "gfx942/fmha_v3_fwd")
    added = [f for f in items if f["kind"] == "row_added_new_value"]
    assert len(added) == 1 and added[0]["tier"] == 2
    assert (
        "hdim_q=256" in added[0]["detail"]
        and "_ZN5aiter8fwd_hd256E" in added[0]["detail"]
    )
    changed = [f for f in items if f["kind"] == "object_changed"]
    assert [f["detail"] for f in changed] == ["MI300/fwd_hd128.co"]


def test_exclude_rule_matches_every_column_it_names(sync, env):
    """dtype=bf16;bf16_cvt=1 must drop the bf16 RTNA row but keep the fp8 row,
    which also carries bf16_cvt=1."""
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.1.0",
        "--arch",
        "gfx942",
        "--exclude",
        "gfx942/fmha_v3_fwd",
        "dtype=bf16;bf16_cvt=1",
        "RTNE only",
        "--report",
        str(report),
    )
    plan = json.loads(report.read_text())["modules"]["gfx942/fmha_v3_fwd"]
    assert [r["knl_name"] for r in plan["excluded_rows"]] == [
        "_ZN5aiter13fwd_hd128_rtnaE"
    ]
    assert "MI300/fwd_fp8.co" in plan["target"]


def test_upstream_abi_and_dispatch_changes_are_diffed(sync, env):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(
        kernels,
        "gfx942",
        [ROW_HD128, ROW_HD128_RTNA, ROW_FP8],
        roots=("MI300", "MI308"),
    )
    aiter.write(
        "csrc/include/mha_fwd.h",
        MHA_FWD_H.replace("p2 _p0;", "p2 _p0;\n    int nsplit;"),
    )
    aiter.write(
        "csrc/cpp_itfs/mha_fwd.cu",
        MHA_FWD_CU.replace(
            "if(a.hdim_q != 128)", "if(a.hdim_q != 128 || a.nhead_q != a.nhead_k)"
        ),
    )
    aiter.commit("v0.2.0")

    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.2.0",
        "--arch",
        "gfx942",
        "--baseline",
        base,
        "--report",
        str(report),
    )
    changed = {
        f["detail"].split(":")[0] + ":" + f["detail"].split(":")[2]: f
        for f in findings(report, "gfx942/fmha_v3_fwd")
        if f["kind"] == "reference_changed"
    }
    struct_change = changed["csrc/include/mha_fwd.h:fmha_fwd_v3_args"]
    dispatch_change = changed["csrc/cpp_itfs/mha_fwd.cu:fmha_fwd_v3"]
    assert struct_change["tier"] == 3 and "+    int nsplit;" in struct_change["diff"]
    assert (
        dispatch_change["tier"] == 2
        and "nhead_q != a.nhead_k" in dispatch_change["diff"]
    )
    assert len(changed) == 2  # the declaration and fmha_fwd_ck() are not compared


def test_ck_objects_block_apply(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.module(
        "gfx942", [ROW_HD128], roots=("MI300", "MI308"), ck={"_ZN5aiter8fwd_hd128E"}
    )
    aiter.commit("v0.2.0")
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.2.0",
        "--arch",
        "gfx942",
        "--report",
        str(report),
    )
    assert "ck_symbol" in kinds(report, "gfx942/fmha_v3_fwd")
    assert run(sync, tree, aiter, cache, "apply", "--report", str(report)) == 2
    assert "blocking findings" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


def test_apply_refuses_a_tree_changed_since_the_plan(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128], roots=("MI300", "MI308"))
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.1.0",
        "--arch",
        "gfx942",
        "--baseline",
        base,
        "--report",
        str(report),
    )
    (kernels / "gfx942/fmha_v3_fwd/MI300/fwd_hd128.co").write_bytes(b"edited")
    assert run(sync, tree, aiter, cache, "apply", "--report", str(report)) == 2
    assert "changed since the plan" in capsys.readouterr().err


def test_apply_never_writes_outside_tracked_forward_modules(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.1.0",
        "--arch",
        "gfx942",
        "--report",
        str(report),
    )
    doctored = json.loads(report.read_text())
    doctored["modules"]["gfx942/fmha_v3_bwd"] = doctored["modules"].pop(
        "gfx942/fmha_v3_fwd"
    )
    report.write_text(json.dumps(doctored))
    assert run(sync, tree, aiter, cache, "apply", "--report", str(report)) == 2
    assert "not a tracked forward ASM module" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


def test_verify_catches_tampering(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.1.0",
        "--arch",
        "gfx942",
        "--report",
        str(report),
    )
    run(sync, tree, aiter, cache, "apply", "--report", str(report))
    module = kernels / "gfx942/fmha_v3_fwd"
    (module / "MI308/fwd_fp8.co").write_bytes(kernel(ROW_FP8, "patched"))
    (module / "MI300/stray.co").write_bytes(kernel(ROW_HD256))
    source = module / "SOURCE.md"
    source.write_text(source.read_text().replace("v0.1.0", "v9.9.9"))
    capsys.readouterr()
    assert run(sync, tree, aiter, cache, "verify") == 1
    out = capsys.readouterr().out
    assert "MI308/fwd_fp8.co: SHA256 differs from the manifest" in out
    assert "MI300/stray.co: on disk, not in the manifest" in out
    assert "MI300/stray.co: no CSV row references it" in out
    assert "SOURCE.md: does not match the pin" in out


def test_new_arch_is_structural_and_archs_lists_only_ported_modules(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.write(
        "hsa/gfx1250/fmha_fwd_bf16/fmha_fwd_bf16.csv",
        "dtype,hdim_q,hdim_v,mask,knl_name,co_name\n"
        "bf16,128,128,0,_ZN5aiter4fwd1E,fwd1.co\n",
    )
    aiter.write("hsa/gfx1250/fmha_fwd_bf16/fwd1.co", elf("_ZN5aiter4fwd1E"))
    aiter.write("csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu", "struct KernelArgs {};\n")
    aiter.write("hsa/gfx1250/fmha_fwd_mxfp8/fmha_fwd_mxfp8.csv", "knl_name,co_name\n")
    aiter.write("hsa/gfx90a/other/x.co", b"x")
    aiter.commit("v0.2.0")

    capsys.readouterr()
    assert run(sync, tree, aiter, cache, "archs", "--ref", "v0.2.0", "--json") == 0
    listed = {a["arch"]: a for a in json.loads(capsys.readouterr().out)["archs"]}
    assert set(listed) == {"gfx942", "gfx1250"}
    assert listed["gfx1250"]["tracked"] == ["fmha_fwd_bf16"]
    assert listed["gfx1250"]["untracked"] == ["fmha_fwd_mxfp8"]

    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.2.0",
        "--arch",
        "gfx1250",
        "--report",
        str(report),
    )
    assert {
        "new_arch",
        "new_module",
        "cmake_arch_missing",
        "cmake_module_missing",
    } <= kinds(report, "gfx1250/fmha_fwd_bf16")
    assert json.loads(report.read_text())["tier"] == 3


def test_latest_without_github_takes_the_highest_release_tag(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    for tag in ("v0.1.9", "v0.1.10", "v0.1.10.post1", "v0.2.0rc1"):
        git(aiter.root, "tag", tag)
    capsys.readouterr()
    assert run(sync, tree, aiter, cache, "latest") == 0
    assert capsys.readouterr().out.startswith("v0.1.10.post1 ")


def test_extract_region_skips_declarations(sync):
    ref = sync.Ref("x.cu", "function", "fmha_fwd_v3", 2, "dispatch")
    region = sync.extract_region(MHA_FWD_CU, ref)
    assert region.startswith(
        "float fmha_fwd_v3(mha_fwd_args a, const stream_config& s)\n{"
    )
    assert "return -1" in region and "fmha_fwd_ck" not in region


def test_host_code_baseline_lags_until_marked_ported(sync, env):
    """Applying new objects must not hide an unported ABI change from later plans."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "csrc/include/aiter_hip_common.h",
        HIP_COMMON_H.replace("_p0, _p1;", "_p0, _p1, _x;"),
    )
    target = aiter.commit("v0.2.0")
    plan_args = ("plan", "--ref", "v0.2.0", "--arch", "gfx942", "--report", str(report))

    run(sync, tree, aiter, cache, *plan_args, "--baseline", base)
    run(sync, tree, aiter, cache, "apply", "--report", str(report))
    pin = json.loads((kernels / "aiter_fwd_pins.json").read_text())["modules"]
    assert pin["gfx942/fmha_v3_fwd"]["abi_commit"] == base

    run(sync, tree, aiter, cache, *plan_args)
    assert {"reference_changed", "abi_unported"} <= kinds(report, "gfx942/fmha_v3_fwd")

    assert (
        run(sync, tree, aiter, cache, "mark-ported", "--module", "gfx942/fmha_v3_fwd")
        == 0
    )
    assert run(sync, tree, aiter, cache, "verify") == 0
    run(sync, tree, aiter, cache, *plan_args)
    assert not {"reference_changed", "abi_unported"} & kinds(
        report, "gfx942/fmha_v3_fwd"
    )
    pin = json.loads((kernels / "aiter_fwd_pins.json").read_text())["modules"]
    assert pin["gfx942/fmha_v3_fwd"]["abi_commit"] == target


def test_known_values_in_a_new_combination_need_review(sync, env):
    """fp8 and group mode are each served, but fp8 in group mode is not."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(
        kernels,
        "gfx942",
        [ROW_HD128, ROW_FP8, ROW_HD128_GROUP],
        roots=("MI300", "MI308"),
    )
    aiter.module(
        "gfx942",
        [ROW_HD128, ROW_FP8, ROW_HD128_GROUP, ROW_FP8_GROUP],
        roots=("MI300", "MI308"),
    )
    aiter.commit("v0.2.0")
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.2.0",
        "--arch",
        "gfx942",
        "--baseline",
        base,
        "--report",
        str(report),
    )
    added = [
        f for f in findings(report, "gfx942/fmha_v3_fwd") if "fwd_fp8g" in f["detail"]
    ]
    assert [(f["kind"], f["tier"]) for f in added] == [("row_added_new_combination", 2)]


def test_pinned_exclusions_are_deduplicated_and_droppable(sync, env):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    rule = ("--exclude", "gfx942/fmha_v3_fwd", "dtype=bf16;bf16_cvt=1", "RTNE only")
    plan_args = ("plan", "--ref", "v0.1.0", "--arch", "gfx942", "--report", str(report))
    run(sync, tree, aiter, cache, *plan_args, *rule)
    run(sync, tree, aiter, cache, "apply", "--report", str(report))

    run(sync, tree, aiter, cache, *plan_args, *rule)
    assert (
        len(json.loads(report.read_text())["modules"]["gfx942/fmha_v3_fwd"]["exclude"])
        == 1
    )

    run(
        sync,
        tree,
        aiter,
        cache,
        *plan_args,
        "--drop-exclude",
        "gfx942/fmha_v3_fwd",
        "dtype=bf16;bf16_cvt=1",
    )
    plan = json.loads(report.read_text())["modules"]["gfx942/fmha_v3_fwd"]
    assert plan["exclude"] == [] and plan["excluded_rows"] == []
    assert "MI300/fwd_hd128_rtna.co" in plan["target"]


def test_symbol_table_past_the_end_is_reported_not_crashed_on(sync, env):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    broken = bytearray(kernel(ROW_FP8))
    symtab_header = len(broken) - 2 * 64  # the third of four section headers
    struct.pack_into("<Q", broken, symtab_header + 32, 1 << 20)  # sh_size
    aiter.write("hsa/gfx942/fmha_v3_fwd/MI300/fwd_fp8.co", bytes(broken))
    aiter.commit("v0.2.0")
    assert (
        run(
            sync,
            tree,
            aiter,
            cache,
            "plan",
            "--ref",
            "v0.2.0",
            "--arch",
            "gfx942",
            "--report",
            str(report),
        )
        == 0
    )
    not_elf = [
        f for f in findings(report, "gfx942/fmha_v3_fwd") if f["kind"] == "not_elf"
    ]
    assert [f["detail"] for f in not_elf] == [
        "MI300/fwd_fp8.co: symbol or string table runs past the end of the file"
    ]


def test_apply_refuses_a_report_that_escapes_the_module(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    run(
        sync,
        tree,
        aiter,
        cache,
        "plan",
        "--ref",
        "v0.1.0",
        "--arch",
        "gfx942",
        "--report",
        str(report),
    )
    doctored = json.loads(report.read_text())
    doctored["modules"]["gfx942/fmha_v3_fwd"]["target"]["../../escape.co"] = "0" * 64
    report.write_text(json.dumps(doctored))
    assert run(sync, tree, aiter, cache, "apply", "--report", str(report)) == 2
    assert "does not write" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()

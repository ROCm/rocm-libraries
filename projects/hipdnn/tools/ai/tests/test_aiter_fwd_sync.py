# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-aiter-asm-sdpa-fwd-sync skill's aiter_fwd_sync.py.

Each test builds a small AITER-shaped git repository (two tagged commits) and a
rocm-libraries-shaped tree in tmp_path, then drives the script's CLI. Objects are
minimal ELF files carrying only a symbol table, which is all the script reads.
No test touches the network or the real checkout.
"""

import dataclasses
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
KEY = "gfx942/fmha_v3_fwd"

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
static bool is_mi308_device()
{
    return false;
}
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

# AITER's Python entry points each define their own can_impl_fmha_v3_fwd(), which
# calls a nested is_fmha_v3_fp8() of the same caller and the module-level extent
# helper. _fmha_bwd_extent_ge_u32() is defined but not called by either gate.
MHA_PY = """\
def _fmha_kv_byte_extent_ge_u32(max_seqlen_k, k, v):
    return max_seqlen_k * k.stride(-3) >= (1 << 32)


def _fmha_bwd_extent_ge_u32(max_seqlen_k, k):
    return max_seqlen_k * k.stride(-2) >= (1 << 32)


def _flash_attn_forward(q, k, v, hdim_q):
    def is_fmha_v3_fp8():
        return q.dtype == "fp8" and hdim_q == 128

    def can_impl_fmha_v3_fwd():
        ret = hdim_q == 128
        ret = ret or is_fmha_v3_fp8()
        ret = ret and not _fmha_kv_byte_extent_ge_u32(4096, k, v)
        return ret

    if can_impl_fmha_v3_fwd():
        return fmha_v3_fwd(q, k, v)
    return mha_fwd(q, k, v)


def _flash_attn_varlen_forward(q, k, v, cu_seqlens_q):
    def is_fmha_v3_fp8():
        return q.dtype == "fp8"

    def can_impl_fmha_v3_fwd():
        ret = cu_seqlens_q is not None
        ret = ret or is_fmha_v3_fp8()
        ret = ret and not _fmha_kv_byte_extent_ge_u32(4096, k, v)
        return ret

    if can_impl_fmha_v3_fwd():
        return fmha_v3_varlen_fwd(q, k, v)
    return mha_varlen_fwd(q, k, v)
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


def csv_text(*rows: str, crlf: bool = False, header: str = HEADER) -> str:
    newline = "\r\n" if crlf else "\n"
    return newline.join([header, *rows]) + newline


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


def run(sync, env, *args: str) -> int:
    aiter, tree, _, cache, _ = env
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


def plan(sync, env, ref: str, *extra: str, arch: str = "gfx942") -> int:
    report = env[4]
    return run(
        sync, env, "plan", "--ref", ref, "--arch", arch, *extra, "--report", str(report)
    )


def apply(sync, env) -> int:
    return run(sync, env, "apply", "--report", str(env[4]))


def base_upstream(aiter: Aiter) -> str:
    aiter.write("csrc/include/mha_fwd.h", MHA_FWD_H)
    aiter.write("csrc/include/aiter_hip_common.h", HIP_COMMON_H)
    aiter.write("csrc/cpp_itfs/mha_fwd.cu", MHA_FWD_CU)
    aiter.write("aiter/ops/mha.py", MHA_PY)
    aiter.write("hsa/codegen.py", "# codegen\n")
    aiter.module(
        "gfx942", [ROW_HD128, ROW_HD128_RTNA, ROW_FP8], roots=("MI300", "MI308")
    )
    return aiter.commit("v0.1.0")


def module_plan(report: Path, key: str = KEY) -> dict:
    return json.loads(report.read_text())["modules"][key]


def findings(report: Path, key: str = KEY) -> list[dict]:
    return module_plan(report, key)["findings"]


def kinds(report: Path, key: str = KEY) -> set[str]:
    return {f["kind"] for f in findings(report, key)}


def of_kind(report: Path, kind: str, key: str = KEY) -> list[dict]:
    return [f for f in findings(report, key) if f["kind"] == kind]


def pin(kernels: Path, key: str = KEY) -> dict:
    return json.loads((kernels / "aiter_pins.json").read_text())["modules"][key]


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

    assert run(sync, env, "verify") == 1  # unpinned before the first update
    assert plan(sync, env, "v0.2.0", "--baseline", base) == 0
    planned = module_plan(report)
    assert planned["tier"] == 0, planned["findings"]
    assert [r["knl_name"] for r in planned["excluded_rows"]] == [
        "_ZN5aiter13fwd_hd128_rtnaE"
    ]

    assert apply(sync, env) == 0
    assert run(sync, env, "verify") == 0
    module = kernels / "gfx942/fmha_v3_fwd"
    assert not (module / "MI300/fwd_hd128_rtna.co").exists()
    assert (module / "fmha_fwd.csv").read_text() == csv_text(ROW_HD128, ROW_FP8)
    assert pin(kernels)["tag"] == "v0.2.0"
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

    plan(sync, env, "v0.2.0", "--baseline", base)
    added = of_kind(report, "row_added_new_value")
    assert len(added) == 1 and added[0]["tier"] == 2
    assert (
        "hdim_q=256" in added[0]["detail"]
        and "_ZN5aiter8fwd_hd256E" in added[0]["detail"]
    )
    changed = of_kind(report, "object_changed")
    assert [f["detail"] for f in changed] == ["MI300/fwd_hd128.co"]


@pytest.mark.parametrize(
    "changed_row, tier, columns",
    [
        (ROW_HD128.replace("128,0,0,0,", "128,1,0,0,"), 2, "mask"),
        (ROW_HD128.replace("256,32,", "256,64,"), 1, "ts_kv"),
    ],
    ids=["dispatch-column", "tile-size-only"],
)
def test_row_change_tier_follows_the_columns_that_changed(
    sync, env, changed_row, tier, columns
):
    """A dispatch column such as mask changes which graphs the kernel serves; a
    tile size changes only how the engine launches it."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv",
        csv_text(changed_row, ROW_HD128_RTNA, ROW_FP8),
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    assert [(f["tier"], f["detail"]) for f in of_kind(report, "row_changed")] == [
        (tier, f"_ZN5aiter8fwd_hd128E: {columns}")
    ]


def test_upstream_dropping_a_vendored_row_is_reported(sync, env):
    """A kernel AITER stops shipping is lost ASM coverage the engine must decline."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv", csv_text(ROW_HD128, ROW_HD128_RTNA)
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    removed = of_kind(report, "row_removed_upstream")
    assert [(f["tier"], f["detail"].split(":")[0]) for f in removed] == [
        (2, "_ZN5aiter6fwd_fp8E")
    ]


def test_vendored_header_change_is_structural(sync, env):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv",
        csv_text(
            ROW_HD128,
            ROW_HD128_RTNA,
            ROW_FP8,
            header=HEADER.replace("bf16_cvt", "cvt"),
        ),
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    assert [f["tier"] for f in of_kind(report, "header_changed")] == [3]


def test_object_directory_change_is_structural(sync, env):
    """The engine picks MI300/ or MI308/ objects per device; losing one is a
    layout change, not a set of removed objects."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    for row in (ROW_HD128, ROW_HD128_RTNA, ROW_FP8):
        aiter.remove(f"hsa/gfx942/fmha_v3_fwd/MI308/{row.split(',')[-1]}")
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    assert [f["tier"] for f in of_kind(report, "layout_changed")] == [3]


def test_exclude_rule_matches_every_column_it_names(sync, env):
    """dtype=bf16;bf16_cvt=1 must drop the bf16 RTNA row but keep the fp8 row,
    which also carries bf16_cvt=1."""
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(
        sync,
        env,
        "v0.1.0",
        "--exclude",
        KEY,
        "dtype=bf16;bf16_cvt=1",
        "RTNE only",
    )
    planned = module_plan(report)
    assert [r["knl_name"] for r in planned["excluded_rows"]] == [
        "_ZN5aiter13fwd_hd128_rtnaE"
    ]
    assert "MI300/fwd_fp8.co" in planned["target"]


def test_apply_deletes_the_objects_of_a_row_a_new_rule_drops(sync, env):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    apply(sync, env)

    plan(sync, env, "v0.1.0", "--exclude", KEY, "dtype=fp8bf16", "no fp8")
    assert [f["tier"] for f in of_kind(report, "row_dropped_by_rule")] == [2]
    assert apply(sync, env) == 0
    module = kernels / "gfx942/fmha_v3_fwd"
    assert not (module / "MI300/fwd_fp8.co").exists()
    assert not (module / "MI308/fwd_fp8.co").exists()
    assert run(sync, env, "verify") == 0


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

    plan(sync, env, "v0.2.0", "--baseline", base)
    changed = {
        f["detail"].split(":")[0] + ":" + f["detail"].split(":")[2]: f
        for f in of_kind(report, "reference_changed")
    }
    struct_change = changed["csrc/include/mha_fwd.h:fmha_fwd_v3_args"]
    dispatch_change = changed["csrc/cpp_itfs/mha_fwd.cu:fmha_fwd_v3"]
    assert struct_change["tier"] == 3 and "+    int nsplit;" in struct_change["diff"]
    assert (
        dispatch_change["tier"] == 3
        and "nhead_q != a.nhead_k" in dispatch_change["diff"]
    )
    assert len(changed) == 2  # the declaration and fmha_fwd_ck() are not compared


@pytest.mark.parametrize(
    "caller, gate_change, outside_change",
    [
        (
            "_flash_attn_forward",
            ("ret = hdim_q == 128", "ret = hdim_q in (128, 192)"),
            ("return mha_fwd(", "return mha_fwd_ck("),
        ),
        (
            "_flash_attn_varlen_forward",
            ("ret = cu_seqlens_q is not None", "ret = cu_seqlens_q is None"),
            ("return mha_varlen_fwd(", "return mha_varlen_fwd_ck("),
        ),
    ],
    ids=["batch", "varlen"],
)
def test_python_gate_change_in_either_copy_is_reported(
    sync, env, caller, gate_change, outside_change
):
    """The batch and varlen entry points each define can_impl_fmha_v3_fwd(); a
    change to one is reported under its caller and the rest of the caller is not
    compared."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "aiter/ops/mha.py",
        MHA_PY.replace(*gate_change).replace(*outside_change),
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    changed = of_kind(report, "reference_changed")
    assert [(f["tier"], f["detail"].split(": ")[0]) for f in changed] == [
        (3, f"aiter/ops/mha.py::{caller}.can_impl_fmha_v3_fwd")
    ]
    assert f"+        {gate_change[1]}" in changed[0]["diff"]
    assert outside_change[1] not in changed[0]["diff"]


@pytest.mark.parametrize(
    "helper_change, callers",
    [
        (
            ('"fp8" and hdim_q == 128', '"fp8" and hdim_q in (128, 256)'),
            ["_flash_attn_forward"],
        ),
        (
            ("k.stride(-3) >= (1 << 32)", "k.stride(-3) >= (1 << 31)"),
            ["_flash_attn_forward", "_flash_attn_varlen_forward"],
        ),
        (("k.stride(-2) >= (1 << 32)", "k.stride(-2) >= (1 << 31)"), []),
    ],
    ids=["nested-helper", "module-helper", "uncalled-function"],
)
def test_python_gate_reports_a_change_confined_to_a_helper_it_calls(
    sync, env, helper_change, callers
):
    """A gate's verdict depends on the helpers it calls. A change to one is reported
    under every gate that calls it, apply keeps the host-code commit, and a change
    to a function no gate calls is not reported."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write("aiter/ops/mha.py", MHA_PY.replace(*helper_change))
    target = aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    changed = of_kind(report, "reference_changed")
    assert [f["detail"].split(": ")[0] for f in changed] == [
        f"aiter/ops/mha.py::{caller}.can_impl_fmha_v3_fwd" for caller in callers
    ]
    for finding in changed:
        added = [line for line in finding["diff"].splitlines() if line[:1] == "+"]
        assert any(helper_change[1] in line for line in added)
    assert apply(sync, env) == 0
    assert pin(kernels)["abi_commit"] == (base if callers else target)


def test_host_code_change_names_the_archs_the_plan_leaves_alone(sync, env):
    """The engine launches every arch's objects with one copy of the mirrored host
    code, so a reference change planned for one arch reaches the others."""
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.module("gfx950", [ROW_HD128])
    base = aiter.commit("v0.1.1")
    vendor(kernels, "gfx942", [ROW_HD128], roots=("MI300", "MI308"))
    vendor(kernels, "gfx950", [ROW_HD128])
    aiter.write(
        "csrc/include/mha_fwd.h",
        MHA_FWD_H.replace("p2 _p0;", "p2 _p0;\n    int nsplit;"),
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base, arch="gfx950")
    shared = of_kind(
        report, "host_code_shared_with_unplanned_archs", "gfx950/fmha_v3_fwd"
    )
    assert [(f["tier"], "gfx942" in f["detail"]) for f in shared] == [(3, True)]

    plan(sync, env, "v0.2.0", "--baseline", base, arch="gfx942,gfx950")
    for key in ("gfx942/fmha_v3_fwd", "gfx950/fmha_v3_fwd"):
        assert "reference_changed" in kinds(report, key)
        assert "host_code_shared_with_unplanned_archs" not in kinds(report, key)


def test_renamed_reference_keeps_the_host_code_baseline(sync, env, monkeypatch):
    """A mirrored function renamed upstream is missing under its old name. apply
    must keep the old host-code commit, so that once REFERENCES names the new
    function the next plan still shows what changed in it."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "csrc/cpp_itfs/mha_fwd.cu",
        MHA_FWD_CU.replace("get_grid_dim(", "compute_grid(").replace(
            "return {1, 1, 1};", "return {2, 1, 1};"
        ),
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    assert [f["tier"] for f in of_kind(report, "reference_missing")] == [3]
    assert apply(sync, env) == 0
    assert pin(kernels)["abi_commit"] == base

    monkeypatch.setitem(
        sync.REFERENCES,
        "fmha_v3_fwd",
        tuple(
            (
                dataclasses.replace(ref, name="compute_grid")
                if ref.name == "get_grid_dim"
                else ref
            )
            for ref in sync.REFERENCES["fmha_v3_fwd"]
        ),
    )
    plan(sync, env, "v0.2.0")
    changed = of_kind(report, "reference_changed")
    assert [f["detail"].split(": ")[0] for f in changed] == [
        "csrc/cpp_itfs/mha_fwd.cu::compute_grid"
    ]
    assert "+    return {2, 1, 1};" in changed[0]["diff"]


def test_ck_objects_block_apply(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.module(
        "gfx942", [ROW_HD128], roots=("MI300", "MI308"), ck={"_ZN5aiter8fwd_hd128E"}
    )
    aiter.commit("v0.2.0")
    plan(sync, env, "v0.2.0")
    assert "ck_symbol" in kinds(report)
    assert apply(sync, env) == 2
    assert "blocking findings" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


def test_object_without_its_kernel_symbol_blocks_apply(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.write("hsa/gfx942/fmha_v3_fwd/MI300/fwd_fp8.co", elf("_ZN5aiter5otherE"))
    aiter.commit("v0.2.0")
    plan(sync, env, "v0.2.0")
    assert [f["detail"] for f in of_kind(report, "symbol_missing")] == [
        "MI300/fwd_fp8.co: does not define _ZN5aiter6fwd_fp8E"
    ]
    assert apply(sync, env) == 2
    assert "blocking findings" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


def test_apply_refuses_a_tree_changed_since_the_plan(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128], roots=("MI300", "MI308"))
    plan(sync, env, "v0.1.0", "--baseline", base)
    (kernels / "gfx942/fmha_v3_fwd/MI300/fwd_hd128.co").write_bytes(b"edited")
    assert apply(sync, env) == 2
    assert "changed since the plan" in capsys.readouterr().err


def test_apply_never_writes_outside_tracked_modules(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    doctored = json.loads(report.read_text())
    doctored["modules"]["gfx942/fmha_v4_fwd"] = doctored["modules"].pop(KEY)
    report.write_text(json.dumps(doctored))
    assert apply(sync, env) == 2
    assert "not a tracked ASM module" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


def test_apply_with_an_untracked_pin_rewrites_nothing(sync, env, capsys):
    """A pin naming a module this tool does not write fails apply before any
    module file is rewritten, so the tree still matches its pin."""
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    assert apply(sync, env) == 0
    pins_path = kernels / "aiter_pins.json"
    pins = json.loads(pins_path.read_text())
    pins["modules"]["gfx942/fmha_v4_fwd"] = pins["modules"][KEY]
    pins_path.write_text(json.dumps(pins, indent=2) + "\n")
    aiter.write("hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128.co", kernel(ROW_HD128, "v2"))
    aiter.commit("v0.2.0")
    assert plan(sync, env, "v0.2.0") == 0

    def snapshot() -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in kernels.rglob("*") if p.is_file()}

    before = snapshot()
    capsys.readouterr()
    assert apply(sync, env) == 2
    assert "not a tracked ASM module" in capsys.readouterr().err
    assert snapshot() == before


def test_verify_catches_tampering(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    apply(sync, env)
    module = kernels / "gfx942/fmha_v3_fwd"
    (module / "MI308/fwd_fp8.co").write_bytes(kernel(ROW_FP8, "patched"))
    (module / "MI300/stray.co").write_bytes(kernel(ROW_HD256))
    source = module / "SOURCE.md"
    source.write_text(source.read_text().replace("v0.1.0", "v9.9.9"))
    capsys.readouterr()
    assert run(sync, env, "verify") == 1
    out = capsys.readouterr().out
    assert "MI308/fwd_fp8.co: SHA256 differs from the manifest" in out
    assert "MI300/stray.co: on disk, not in the manifest" in out
    assert "MI300/stray.co: no CSV row references it" in out
    assert "SOURCE.md: does not match the pin" in out


FP8_KNL = ROW_FP8.split(",")[-2]


def _plant_object(*symbols: str):
    def plant(module: Path, kernels: Path) -> None:
        (module / "MI308/fwd_fp8.co").write_bytes(elf(*symbols))

    return plant


def _delete_object(module: Path, kernels: Path) -> None:
    (module / "MI308/fwd_fp8.co").unlink()


def _exclude_a_kept_row(module: Path, kernels: Path) -> None:
    path = kernels / "aiter_pins.json"
    pins = json.loads(path.read_text())
    pins["modules"][KEY]["exclude"].append(
        {"match": {"dtype": ["fp8bf16"]}, "reason": "no fp8"}
    )
    path.write_text(json.dumps(pins))


@pytest.mark.parametrize(
    "plant, message",
    [
        (
            _plant_object("_ZN5aiter5otherE"),
            f"MI308/fwd_fp8.co: does not define {FP8_KNL}",
        ),
        *(
            (
                _plant_object(FP8_KNL, ck),
                f"MI308/fwd_fp8.co: CK symbols ['{ck}']",
            )
            # One symbol per CK_SYMBOL alternative, each matching no other.
            for ck in ("_ZN2ck6kernelE", "_ZNK2ck3runEv", "my_ck_tile_fn", "kentry")
        ),
        (
            _delete_object,
            f"fmha_fwd.csv: {FP8_KNL} names MI308/fwd_fp8.co, which is missing",
        ),
        (
            _exclude_a_kept_row,
            f"fmha_fwd.csv: {FP8_KNL} is kept but an exclusion matches it",
        ),
    ],
    ids=[
        "no-kernel-symbol",
        "ck-namespace",
        "ck-const-member",
        "ck_tile",
        "kentry",
        "object-missing",
        "excluded-row-kept",
    ],
)
def test_verify_reports_each_defect_planted_after_apply(
    sync, env, capsys, plant, message
):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    apply(sync, env)
    plant(kernels / "gfx942/fmha_v3_fwd", kernels)
    capsys.readouterr()
    assert run(sync, env, "verify") == 1
    assert f"FAIL {KEY}/{message}" in capsys.readouterr().out


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
    assert run(sync, env, "archs", "--ref", "v0.2.0", "--json") == 0
    listed = {a["arch"]: a for a in json.loads(capsys.readouterr().out)["archs"]}
    assert set(listed) == {"gfx942", "gfx1250"}
    assert listed["gfx1250"]["tracked"] == ["fmha_fwd_bf16"]
    assert listed["gfx1250"]["untracked"] == ["fmha_fwd_mxfp8"]

    plan(sync, env, "v0.2.0", arch="gfx1250")
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
    assert run(sync, env, "latest") == 0
    assert capsys.readouterr().out.startswith("v0.1.10.post1 ")


def test_extract_region_skips_declarations(sync):
    ref = sync.Ref("x.cu", "function", "fmha_fwd_v3", 2, "dispatch")
    region = sync.extract_region(MHA_FWD_CU, ref)
    assert region.startswith(
        "float fmha_fwd_v3(mha_fwd_args a, const stream_config& s)\n{"
    )
    assert "return -1" in region and "fmha_fwd_ck" not in region


def test_extract_region_keeps_a_return_type_on_its_own_line(sync):
    text = (
        "    auto [y, x] = compute_mask_coordinates(l, r, 0, sq, sk, true);\n"
        "inline std::pair<int, int>\n"
        "compute_mask_coordinates(int l,\n"
        "                         int r)\n"
        "{\n"
        "    return {l, r};\n"
        "}\n"
    )
    ref = sync.Ref("x.h", "function", "compute_mask_coordinates", 3, "mask")
    assert sync.extract_region(text, ref) == text[text.index("inline") :].rstrip()


def test_extract_region_reads_python_defs_by_dotted_name(sync):
    """A region is the def by its dotted name plus the functions it calls. A def in
    the function's own body shadows a module-level one of the same name, as AITER's
    _flash_attn_varlen_backward() does with can_impl_fmha_v3_bwd()."""
    gate = (
        "def can_impl_fmha_v3_bwd(\n"
        "    q,\n"
        "    k,\n"
        ") -> bool:\n"
        "    def nested():\n"
        "        return True\n"
        "\n"
        "    return nested() and q > 16\n"
    )
    varlen = (
        "def _flash_attn_varlen_backward(q):\n"
        "    def can_impl_fmha_v3_bwd():\n"
        "        return q > 0\n"
        "\n"
        "    return can_impl_fmha_v3_bwd()\n"
    )
    text = f"{gate}\n\n# trailing comment\ndef other():\n    pass\n\n\n{varlen}"

    def region(name: str) -> str | None:
        ref = sync.Ref("aiter/ops/mha.py", "pyfunction", name, 3, "gate")
        return sync.extract_region(text, ref)

    assert region("can_impl_fmha_v3_bwd") == gate
    assert region("can_impl_fmha_v3_bwd.nested") == (
        "    def nested():\n        return True\n"
    )
    assert region("nested") is None  # a nested def needs its dotted name
    assert region("_flash_attn_varlen_backward") == varlen


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

    plan(sync, env, "v0.2.0", "--baseline", base)
    apply(sync, env)
    assert pin(kernels)["abi_commit"] == base

    plan(sync, env, "v0.2.0")
    assert {"reference_changed", "abi_unported"} <= kinds(report)

    assert run(sync, env, "mark-ported", "--module", KEY) == 0
    assert run(sync, env, "verify") == 0
    plan(sync, env, "v0.2.0")
    assert not {"reference_changed", "abi_unported"} & kinds(report)
    assert pin(kernels)["abi_commit"] == target


def test_codegen_change_stays_reported_until_marked_ported(sync, env):
    """Applying new objects must not hide an unported change to AITER's codegen."""
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write("hsa/codegen.py", "# codegen\nstrip_comments = True\n")
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    apply(sync, env)
    plan(sync, env, "v0.2.0")
    codegen = [
        f
        for f in of_kind(report, "reference_changed")
        if "hsa/codegen.py" in f["detail"]
    ]
    assert len(codegen) == 1 and "+strip_comments = True" in codegen[0]["diff"]

    run(sync, env, "mark-ported", "--module", KEY)
    plan(sync, env, "v0.2.0")
    assert "reference_changed" not in kinds(report)


def test_header_change_on_one_arch_is_structural(sync, env):
    """The vendored codegen needs one header per CSV name across every arch on
    disk, including archs the plan does not update."""
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    aiter.module("gfx950", [ROW_HD128])
    base = aiter.commit("v0.1.1")
    vendor(kernels, "gfx942", [ROW_HD128], roots=("MI300", "MI308"))
    vendor(kernels, "gfx950", [ROW_HD128])
    aiter.write(
        "hsa/gfx950/fmha_v3_fwd/fmha_fwd.csv",
        HEADER.replace("ts_kv,", "ts_kv,flat,")
        + "\n"
        + ROW_HD128.replace("32,_Z", "32,0,_Z")
        + "\n",
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base, arch="gfx950")
    mismatch = of_kind(report, "header_mismatch_across_archs", "gfx950/fmha_v3_fwd")
    assert [(f["tier"], "gfx942" in f["detail"]) for f in mismatch] == [(3, True)]


def test_csv_comments_are_parsed_like_upstream_and_not_vendored(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128, ROW_FP8], roots=("MI300", "MI308"))
    aiter.write(
        "hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv",
        "/* forward kernels\n   for gfx942 */\n"
        + HEADER
        + "\n# bf16\n"
        + ROW_HD128
        + " # round to nearest even\n"
        + ROW_HD128_RTNA
        + "\n"
        + ROW_FP8
        + " // fp8\n",
    )
    aiter.commit("v0.2.0")

    plan(sync, env, "v0.2.0", "--baseline", base)
    assert not {"row_changed", "row_added", "header_changed"} & kinds(report)
    assert apply(sync, env) == 0
    vendored = (kernels / "gfx942/fmha_v3_fwd/fmha_fwd.csv").read_text()
    assert vendored == csv_text(ROW_HD128, ROW_FP8)
    assert run(sync, env, "verify") == 0


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
    plan(sync, env, "v0.2.0", "--baseline", base)
    added = [f for f in findings(report) if "fwd_fp8g" in f["detail"]]
    assert [(f["kind"], f["tier"]) for f in added] == [("row_added_new_combination", 2)]


def test_pinned_exclusions_are_deduplicated_and_droppable(sync, env):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    rule = ("--exclude", KEY, "dtype=bf16;bf16_cvt=1", "RTNE only")
    plan(sync, env, "v0.1.0", *rule)
    apply(sync, env)

    plan(sync, env, "v0.1.0", *rule)
    assert len(module_plan(report)["exclude"]) == 1

    plan(sync, env, "v0.1.0", "--drop-exclude", KEY, "dtype=bf16;bf16_cvt=1")
    planned = module_plan(report)
    assert planned["exclude"] == [] and planned["excluded_rows"] == []
    assert "MI300/fwd_hd128_rtna.co" in planned["target"]


def test_symbol_table_past_the_end_is_reported_not_crashed_on(sync, env):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    broken = bytearray(kernel(ROW_FP8))
    symtab_header = len(broken) - 2 * 64  # the third of four section headers
    struct.pack_into("<Q", broken, symtab_header + 32, 1 << 20)  # sh_size
    aiter.write("hsa/gfx942/fmha_v3_fwd/MI300/fwd_fp8.co", bytes(broken))
    aiter.commit("v0.2.0")
    assert plan(sync, env, "v0.2.0") == 0
    assert [f["detail"] for f in of_kind(report, "not_elf")] == [
        "MI300/fwd_fp8.co: symbol or string table runs past the end of the file"
    ]


def test_apply_refuses_a_report_that_escapes_the_module(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base_upstream(aiter)
    plan(sync, env, "v0.1.0")
    doctored = json.loads(report.read_text())
    doctored["modules"][KEY]["target"]["../../escape.co"] = "0" * 64
    report.write_text(json.dumps(doctored))
    assert apply(sync, env) == 2
    assert "does not write" in capsys.readouterr().err
    assert not (kernels / "gfx942").exists()


BWD_HEADER = "dtype,hdim_q,hdim_v,mask,atomic32,pssk,pddv,mode,bf16_cvt,ts_qo,ts,knl_name,co_name"
ROW_BWD = "bf16,128,128,0,1,1,1,0,3,16,256,_ZN5aiter9bwd_hd128E,bwd_hd128.co"
ROW_SHUFFLE = "bf16,128,128,0,0,0,0,0,3,0,64,_ZN5aiter11bwd_shuffleE,bwd_shuffle.co"


def bwd_upstream(aiter: Aiter) -> None:
    """A backward module: one dqdkdv row, a dq_shuffle table with one row, and a
    CSV that holds only its header."""
    base = "hsa/gfx942/fmha_v3_bwd"
    aiter.write(f"{base}/fmha_bwd_dqdkdv.csv", f"{BWD_HEADER}\n{ROW_BWD}\n")
    aiter.write(f"{base}/fmha_bwd_dq_shuffle.csv", f"{BWD_HEADER}\n{ROW_SHUFFLE}\n")
    aiter.write(f"{base}/fmha_bwd_empty.csv", f"{BWD_HEADER}\n")
    for row in (ROW_BWD, ROW_SHUFFLE):
        aiter.write(f"{base}/{row.split(',')[-1]}", kernel(row))


def test_backward_first_update_keeps_the_vendored_csvs(sync, env, capsys):
    """The first backward update leaves out the CSVs the tree lacked at the
    baseline, takes its own scoped baselines, and keeps a CSV only on --include."""
    aiter, tree, kernels, cache, report = env
    bwd_key = "gfx942/fmha_v3_bwd"
    abi = base_upstream(aiter)
    bwd_upstream(aiter)
    base = aiter.commit("v0.1.1")
    vendor(kernels, "gfx942", [ROW_HD128, ROW_HD128_RTNA, ROW_FP8], ("MI300", "MI308"))
    module = kernels / bwd_key
    module.mkdir()
    (module / "fmha_bwd_dqdkdv.csv").write_text(f"{BWD_HEADER}\n{ROW_BWD}\n")
    for row in (ROW_BWD, ROW_SHUFFLE):  # the dq_shuffle object ships unreferenced
        (module / row.split(",")[-1]).write_bytes(kernel(row))
    baselines = ("--baseline", f"fwd={abi}", "--baseline", f"bwd={base}")
    baselines += ("--abi-baseline", f"bwd={abi}")

    assert plan(sync, env, "v0.1.1", *baselines) == 0
    fwd, bwd = module_plan(report), module_plan(report, bwd_key)
    assert (fwd["baseline"], fwd["abi_baseline"]) == (abi, abi)
    assert (bwd["baseline"], bwd["abi_baseline"]) == (base, abi)
    assert "abi_unported" in kinds(report, bwd_key)
    assert [r["match"] for r in bwd["auto_exclude"]] == [
        {"csv": ["fmha_bwd_dq_shuffle.csv"]},
        {"csv": ["fmha_bwd_empty.csv"]},
    ]
    assert sorted(p for p in bwd["target"] if p.endswith(".csv")) == [
        "fmha_bwd_dqdkdv.csv"
    ]
    assert ("object_removed", "bwd_shuffle.co") in {
        (f["kind"], f["detail"]) for f in findings(report, bwd_key)
    }

    include = ("--include", bwd_key)
    include = (*include, "fmha_bwd_dq_shuffle.csv", *include, "fmha_bwd_empty.csv")
    assert plan(sync, env, "v0.1.1", *baselines, *include) == 0
    assert "csv_added" in kinds(report, bwd_key)
    assert apply(sync, env) == 0
    assert (module / "fmha_bwd_empty.csv").read_text() == f"{BWD_HEADER}\n"
    assert "ASM backward kernels" in (module / "SOURCE.md").read_text()
    capsys.readouterr()
    assert run(sync, env, "verify", "--direction", "bwd") == 0
    assert f"checked: {bwd_key}\n" in capsys.readouterr().out

    rule = ("--exclude", bwd_key, "csv=fmha_bwd_dqdkdv.csv", "unused")
    assert plan(sync, env, "v0.1.1", "--direction", "bwd", *rule) == 0
    assert "csv_added" not in kinds(report, bwd_key)
    assert "csv_dropped_by_rule" in kinds(report, bwd_key)
    bwd = module_plan(report, bwd_key)
    assert sorted(bwd["target"]) == [
        "bwd_shuffle.co",
        "fmha_bwd_dq_shuffle.csv",
        "fmha_bwd_empty.csv",
    ]


def test_included_csv_keeps_its_rows_despite_a_csv_rule(sync, env, capsys):
    """--include of a whole CSV that a pinned csv rule leaves out keeps its rows
    and objects, and verify accepts the result."""
    aiter, tree, kernels, cache, report = env
    bwd_key = "gfx942/fmha_v3_bwd"
    abi = base_upstream(aiter)
    bwd_upstream(aiter)
    base = aiter.commit("v0.1.1")
    module = kernels / bwd_key
    module.mkdir(parents=True)
    (module / "fmha_bwd_dqdkdv.csv").write_text(f"{BWD_HEADER}\n{ROW_BWD}\n")
    (module / ROW_BWD.split(",")[-1]).write_bytes(kernel(ROW_BWD))
    first = ("--direction", "bwd", "--baseline", f"bwd={base}")
    assert plan(sync, env, "v0.1.1", *first, "--abi-baseline", f"bwd={abi}") == 0
    assert apply(sync, env) == 0  # pins {"csv": ["fmha_bwd_dq_shuffle.csv"]}

    include = ("--include", bwd_key, "fmha_bwd_dq_shuffle.csv")
    assert plan(sync, env, "v0.1.1", "--direction", "bwd", *include) == 0
    assert apply(sync, env) == 0
    assert (module / "fmha_bwd_dq_shuffle.csv").read_text() == (
        f"{BWD_HEADER}\n{ROW_SHUFFLE}\n"
    )
    assert (module / "bwd_shuffle.co").read_bytes() == kernel(ROW_SHUFFLE)
    capsys.readouterr()
    assert run(sync, env, "verify", "--direction", "bwd") == 0


def test_pin_keeps_only_includes_that_override_a_pinned_rule(sync, env):
    """An include that only countered the first update's baseline overrides no
    pinned rule afterwards, so it is not pinned."""
    aiter, tree, kernels, cache, report = env
    bwd_key = "gfx942/fmha_v3_bwd"
    abi = base_upstream(aiter)
    bwd_upstream(aiter)
    base = aiter.commit("v0.1.1")
    module = kernels / bwd_key
    module.mkdir(parents=True)
    (module / "fmha_bwd_dqdkdv.csv").write_text(f"{BWD_HEADER}\n{ROW_BWD}\n")
    (module / ROW_BWD.split(",")[-1]).write_bytes(kernel(ROW_BWD))
    first = ("--direction", "bwd", "--baseline", f"bwd={base}")
    first += ("--abi-baseline", f"bwd={abi}")
    include = ("--include", bwd_key, "fmha_bwd_dq_shuffle.csv")
    assert plan(sync, env, "v0.1.1", *first, *include) == 0
    assert apply(sync, env) == 0
    pin = json.loads((kernels / "aiter_pins.json").read_text())["modules"][bwd_key]
    assert "include" not in pin
    assert (module / "fmha_bwd_dq_shuffle.csv").read_text() == (
        f"{BWD_HEADER}\n{ROW_SHUFFLE}\n"
    )

    rule = ("--exclude", bwd_key, "csv=fmha_bwd_dq_shuffle.csv", "unused")
    assert plan(sync, env, "v0.1.1", "--direction", "bwd", *rule, *include) == 0
    assert apply(sync, env) == 0
    pin = json.loads((kernels / "aiter_pins.json").read_text())["modules"][bwd_key]
    assert pin["include"] == ["fmha_bwd_dq_shuffle.csv"]


def test_baseline_scope_must_name_a_planned_module(sync, env, capsys):
    aiter, tree, kernels, cache, report = env
    base = base_upstream(aiter)
    vendor(kernels, "gfx942", [ROW_HD128], ("MI300", "MI308"))
    assert plan(sync, env, "v0.1.0", "--baseline", f"bwd={base}") == 2
    assert "applies to none of the planned modules" in capsys.readouterr().err

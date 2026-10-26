# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Post-build VGPR budget check for FMHA fwd instances with a residency target.

Codegen declares how many workgroups per CU some gfx1250 instances must keep
resident. One VGPR above the matching budget still compiles but costs a resident
workgroup per CU on hardware, so this test reads the AMDGPU metadata of the built
instance objects and fails when an instance exceeds its budget or spills. No GPU
is required.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_CK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CK_ROOT / "example" / "ck_tile" / "01_fmha"))

from codegen.ops.fmha_fwd import (  # noqa: E402
    KernelComponentFactoryGfx125,
    get_fwd_blobs,
)

# gfx1250 wave32: each SIMD holds 1024 VGPRs, allocated in blocks of 16.
_VGPRS_PER_SIMD = 1024
_VGPR_ALLOC_GRANULE = 16
_KERNEL_RECORD = re.compile(r"^\s+- \.agpr_count:|^\s+- \.args:", re.MULTILINE)
_FIELD = re.compile(
    r"^\s+\.(name|vgpr_count|vgpr_spill_count):\s+(\S+)\s*$", re.MULTILINE
)

_D64_SCHED_OBJECT = re.compile(r"^fmha_fwd_d64_.*_qr_tdm_fp8_sched_.*\.o$")

_ARGS = None


def vgpr_budget(resident_blocks):
    # A workgroup has one wave per SIMD, so blocks per CU equals waves per SIMD.
    granules = _VGPRS_PER_SIMD // resident_blocks // _VGPR_ALLOC_GRANULE
    return granules * _VGPR_ALLOC_GRANULE


def budgeted_kernels(arch):
    _, kernels = get_fwd_blobs([arch], "", 0, [-1], "simplified")
    budgets = {}
    for kernel in kernels:
        blocks = KernelComponentFactoryGfx125.resident_blocks_per_cu(kernel)
        if blocks is not None:
            budgets[kernel.filename + ".o"] = vgpr_budget(blocks)
    return budgets


def llvm_tools(compiler):
    # The HIP compiler may be hipcc (ROCm bin) or clang++ (the LLVM bin itself).
    base = compiler.resolve().parent
    for candidate in (
        base,
        base.parent / "llvm" / "bin",
        base.parent / "lib" / "llvm" / "bin",
    ):
        if (candidate / "clang-offload-bundler").exists():
            return candidate
    raise FileNotFoundError(f"no clang-offload-bundler next to {compiler}")


def kernel_metadata(tools, obj, arch, workdir):
    bundle = workdir / (obj.name + ".fatbin")
    code = workdir / (obj.name + ".co")
    subprocess.run(
        [
            tools / "llvm-objcopy",
            f"--dump-section=.hip_fatbin={bundle}",
            obj,
            "/dev/null",
        ],
        check=True,
    )
    subprocess.run(
        [
            tools / "clang-offload-bundler",
            "--unbundle",
            "--type=o",
            f"--input={bundle}",
            f"--targets=hip-amdgcn-amd-amdhsa--{arch}",
            f"--output={code}",
        ],
        check=True,
    )
    notes = subprocess.run(
        [tools / "llvm-readelf", "--notes", code],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    records = []
    for chunk in _KERNEL_RECORD.split(notes)[1:]:
        fields = dict(_FIELD.findall(chunk))
        if "FmhaFwdKernel" in fields.get("name", ""):
            records.append(fields)
    return records


class VgprBudgetTest(unittest.TestCase):
    def test_budgeted_instances_fit(self):
        objects = {
            Path(line).name: Path(line)
            for line in Path(_ARGS.objects).read_text().split()
            if line.endswith(".o")
        }
        tools = llvm_tools(Path(_ARGS.compiler))
        failures = []
        for arch in _ARGS.archs.split(","):
            budgets = budgeted_kernels(arch)
            present = {n: b for n, b in budgets.items() if n in objects}
            if not present:
                # A build with D64 sched objects but none of the expected names is mismatched.
                self.assertFalse(
                    [n for n in objects if _D64_SCHED_OBJECT.match(n)],
                    f"{arch}: D64 qr_tdm_fp8_sched objects built, but none with a residency "
                    "target; the build's receipt, filter or mask_impl differs from this test",
                )
                continue
            with tempfile.TemporaryDirectory(prefix="fmha-vgpr-budget-") as directory:
                for name, budget in sorted(present.items()):
                    records = kernel_metadata(
                        tools, objects[name], arch, Path(directory)
                    )
                    self.assertTrue(records, f"{name}: no FmhaFwdKernel metadata")
                    for fields in records:
                        vgprs = int(fields["vgpr_count"])
                        spills = int(fields["vgpr_spill_count"])
                        if vgprs > budget or spills:
                            failures.append(
                                f"{arch} {name}: {vgprs} VGPRs ({spills} spilled), "
                                f"budget {budget}"
                            )
        if failures:
            self.fail(
                "instances over the VGPR budget of their residency target "
                "(KernelComponentFactoryGfx125.resident_blocks_per_cu):\n"
                + "\n".join(failures)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument(
        "--archs", required=True, help="comma-separated gfx125x targets of the build"
    )
    parser.add_argument(
        "--objects", required=True, help="file listing the instance objects"
    )
    _ARGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *rest])

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/p_emit_compile.py end to end, on fake parity emitters and a fake clang: target resolution (never a guessed
gfx1250), the end-of-configs rule (a traceback quoting "config_index" is not the end), and new vs old rejections.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "p_emit_compile.py"
PDIR = "platform/tests/instances/parity"

COMMON = """
import sys


def run_emit(spec_fn, build_fn, *, usage=None, arch="gfx950"):
    idx = int(sys.argv[1])
    selected = spec_fn(idx)
    if isinstance(selected, tuple):
        spec, arch = selected
    else:
        spec = selected
    sys.stdout.write(build_fn(spec, arch=arch))
    return 0
"""

# Configs: 0 default arch, 1 pins gfx942 (neither IR names a target), 2 and 3 may be rejected, 4 names gfx1201 in its
# IR, 5 pins a generic target (gfx11-generic).
EMIT = """
from _emit_common import run_emit

REJECT = {reject}


def _spec(idx):
    if idx == 0:
        return "mfma"
    if idx == 1:
        return ("mfma", "gfx942")
    if idx in (2, 3):
        return f"big{{idx}}"
    if idx == 4:
        return "wmma"
    if idx == 5:
        return ("plain", "gfx11-generic")
    raise SystemExit(f"unknown config index {{idx}}")


def build(spec, arch):
    if spec.startswith("big") and int(spec[3:]) in REJECT:
        raise ValueError("invalid spec: LDS budget too big")
    if spec == "wmma":
        return 'define void @k() #0 {{ ret void }}\\nattributes #0 = {{ "target-cpu"="gfx1201" }}\\n'
    if spec == "plain":
        return "define void @k() {{ ret void }}\\n"
    return "; MFMA kernel\\ndefine void @k() {{ ret void }}\\n"


def main():
    return run_emit(
        _spec,
        build,
        usage="usage: fam_emit.py <config_index 0..4>\\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
"""

CLANG = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
cpu = next(a.split("=", 1)[1] for a in args if a.startswith("-mcpu="))
text = open([a for a in args if a.endswith(".ll")][0]).read()
if text.count("target datalayout") > 1:  # clang reads one module per file
    sys.stderr.write("k.ll:6:1: error: expected top-level entity\\n")
    sys.exit(1)
if "MFMA" in text and cpu not in ("gfx942", "gfx950"):
    sys.stderr.write("fatal error: error in backend: Cannot select: intrinsic mfma on " + cpu + "\\n")
    sys.exit(1)
scratch = 16 if "SPILL" in text and "-amdgpu-mfma-vgpr-form=0" not in args else 0
sys.stderr.write(f"remark: ScratchSize [bytes/lane]: {scratch}\\nremark: VGPRs: 4\\n")
"""

# An emitter that lowers kernels with attrs: config 1 is an AGPR body that only compiles clean with rocKE's own
# option, which the tree's helpers.compile._comgr_options_for_kernel gives.
AGPR = """
import sys
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
class K:
    def __init__(self, agpr):
        self.attrs = {"mfma_agpr_form": agpr}
idx = int(sys.argv[1])
if idx > 1:
    raise SystemExit(f"unknown config index {idx}")
sys.stdout.write(lower(K(idx == 1), arch="gfx950"))
"""
LOWER_AGPR = """
def _lower_kernel_to_llvm_python(kernel, arch=None, llvm_flavor=None):
    tag = "; SPILL" if kernel.attrs.get("mfma_agpr_form") else "; plain"
    return tag + "\\ndefine void @k() { ret void }\\n"
"""
OPTIONS = """
def _comgr_options_for_kernel(kernel):
    opts = ["-O3"]
    if kernel.attrs.get("mfma_agpr_form"):
        opts += ["-mllvm", "-amdgpu-mfma-vgpr-form=0"]
    return opts
"""

# Three configs; those in {spill} emit IR the fake clang reports as spilling.
SPILLS = """
import sys
idx = int(sys.argv[1])
if idx > 2:
    raise SystemExit(f"unknown config index {{idx}}")
tag = "; SPILL" if idx in {spill} else "; plain"
sys.stdout.write(tag + '\\ndefine void @k() #0 {{ ret void }}\\nattributes #0 = {{ "target-cpu"="gfx950" }}\\n')
"""


# A family whose emitter refuses every config below llvm23 (gfx1250 tensor loads).
EMIT23 = """
import os, sys
idx = int(sys.argv[1])
if idx > 2:
    raise SystemExit(f"unknown config index {idx}")
if os.environ.get("ROCKE_LLVM_FLAVOR") != "llvm23":
    raise SystemExit("ValueError: tensor_load_to_lds requires LLVM flavor llvm23")
sys.stdout.write('define void @k() #0 { ret void }\\nattributes #0 = { "target-cpu"="gfx1250" }\\n')
"""

# Five configs, run against a config cap of three.
FIVE = """
import sys
idx = int(sys.argv[1])
if idx > 4:
    raise SystemExit(f"unknown config index {idx}")
sys.stdout.write('define void @k() #0 { ret void }\\nattributes #0 = { "target-cpu"="gfx950" }\\n')
"""

# An emitter that lowers directly (no run_emit) into IR that names no target, like conv_direct_grouped_emit.py,
# and the stand-in rocke lowering function it calls.
DIRECT = """
import sys
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
idx = int(sys.argv[1])
if idx > 1:
    raise SystemExit(f"unknown config index {idx}")
sys.stdout.write(lower("kernel", arch="gfx942", llvm_flavor="llvm22"))
"""
LOWER = """
def _lower_kernel_to_llvm_python(kernel, arch=None, llvm_flavor=None):
    return "define void @k() { ret void }\\n"
"""

# Two modules in one emitted file, like gfx950_attention_tiled_3d's segment and reduce kernels.
MULTI = """
import sys
idx = int(sys.argv[1])
if idx > 0:
    raise SystemExit(f"unknown config index {idx}")
mod = 'target datalayout = "e"\\ntarget triple = "amdgcn-amd-amdhsa"\\n\\ndefine void @{n}() #0 {{ ret void }}\\n' \\
      'attributes #0 = {{ "target-cpu"="gfx950" }}\\n'
sys.stdout.write(mod.format(n="segment") + "\\n" + mod.format(n="reduce"))
"""

# Output that is not LLVM IR, like conv_abi's ABI lists.
ABI = """
import sys
idx = int(sys.argv[1])
if idx > 1:
    raise SystemExit(f"unknown config index {idx}")
sys.stdout.write("kernel_abi: a0 a1 a2\\n")
"""

# One config whose output is not LLVM IR, and one both trees refuse, like ir_serialize's outputs and refusals.
SERIAL = """
import sys
idx = int(sys.argv[1])
if idx > 1:
    raise SystemExit(f"unknown config index {idx}")
if idx == 1:
    raise SystemExit("ValueError: refused on purpose")
sys.stdout.write('{"ops": []}\\n')
"""

# One config whose IR the fake clang cannot compile (MFMA on gfx1201), or can (plain).
FAILS = """
import sys
idx = int(sys.argv[1])
if idx > 0:
    raise SystemExit(f"unknown config index {idx}")
sys.stdout.write('; MFMA kernel\\ndefine void @k() #0 { ret void }\\nattributes #0 = { "target-cpu"="gfx1201" }\\n')
"""
COMPILES = FAILS.replace("; MFMA kernel", "; plain kernel")

# A family the PR adds, with a must-reject config.
NEWFAM = """
import sys
idx = int(sys.argv[1])
if idx > 1:
    raise SystemExit(f"unknown config index {idx}")
if idx == 1:
    raise SystemExit("ValueError: must-reject config")
sys.stdout.write('define void @k() #0 { ret void }\\nattributes #0 = { "target-cpu"="gfx1250" }\\n')
"""


class TestEmitCompile(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.d.name)
        for name, reject in (("pr", [2, 3]), ("dev", [3])):
            p = self.base / name / PDIR
            p.mkdir(parents=True)
            (p / "_emit_common.py").write_text(COMMON)
            (p / "fam_emit.py").write_text(EMIT.format(reject=reject))
        b = self.base / "bin"
        b.mkdir()
        (b / "clang").write_text(CLANG)
        os.chmod(b / "clang", 0o755)
        self.env = dict(os.environ, RKMON_LLVM_BIN=str(b), RKMON_TMP=self.d.name)

    def tearDown(self):
        self.d.cleanup()

    def run_tool(self, *extra: str) -> str:
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--root",
                str(self.base / "pr"),
                *extra,
                f"{PDIR}:fam",
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    def test_targets_end_rule_and_new_rejections(self):
        out = self.run_tool("--base-root", str(self.base / "dev"))
        self.assertTrue(
            out.startswith(
                "fam: 4 compiled, 0 compile failures, 1 newly rejected, "
                "1 rejected on develop too, spills 0"
            ),
            out,
        )
        self.assertIn(
            "cfg 2: NEW rejection (develop builds it): ValueError: invalid spec", out
        )

    def test_without_a_develop_tree_rejections_are_only_counted(self):
        out = self.run_tool()
        self.assertTrue(
            out.startswith(
                "fam: 4 compiled, 0 compile failures, 0 newly rejected, 0 rejected on develop "
                "too, 2 rejected (not compared with develop)"
            ),
            out,
        )

    def test_an_emitter_that_lowers_directly_reports_its_target(self):
        """An emitter that calls the lowering itself (conv_direct_grouped_emit.py) writes IR that names no target; its
        configs still compile, never "unknown target"."""
        root = self.base / "pr"
        (root / PDIR / "direct_emit.py").write_text(DIRECT)
        core = root / "platform/python/rocke/core"
        core.mkdir(parents=True)
        (root / "platform/python/rocke/__init__.py").write_text("")
        (core / "__init__.py").write_text("")
        (core / "lower_llvm.py").write_text(LOWER)
        p = subprocess.run(
            [sys.executable, "-B", str(TOOL), "--root", str(root), f"{PDIR}:direct"],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        self.assertTrue(
            p.stdout.startswith("direct: 2 compiled, 0 compile failures"),
            p.stdout + p.stderr,
        )

    def test_a_family_that_needs_another_flavor_is_compiled_at_it(self):
        """An llvm23-only family at llvm22 does not count every config as rejected and pass with 0 compiled."""
        for name in ("pr", "dev"):
            (self.base / name / PDIR / "f23_emit.py").write_text(EMIT23)
        env = dict(self.env, ROCKE_LLVM_FLAVOR="llvm22")
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--root",
                str(self.base / "pr"),
                "--base-root",
                str(self.base / "dev"),
                f"{PDIR}:f23",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertTrue(
            p.stdout.startswith(
                "f23: 3 compiled, 0 compile failures, 0 newly rejected"
            ),
            p.stdout + p.stderr,
        )

    def tool(self, fam: str, src: str, base: bool = True) -> str:
        (self.base / "pr" / PDIR / f"{fam}_emit.py").write_text(src)
        extra = ["--base-root", str(self.base / "dev")] if base else []
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--root",
                str(self.base / "pr"),
                *extra,
                f"{PDIR}:{fam}",
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        return p.stdout + p.stderr

    def test_each_module_of_a_file_is_compiled_on_its_own(self):
        """Each module of a file (gfx950_attention_tiled_3d's reduce kernel) compiles on its own, never "expected
        top-level entity"."""
        out = self.tool("multi", MULTI)
        self.assertTrue(out.startswith("multi: 1 compiled, 0 compile failures"), out)

    def test_output_that_is_not_ir_is_not_a_family_with_nothing_compiled(self):
        """conv_abi's ABI lists do not fail G7 as "nothing compiled"."""
        out = self.tool("abi", ABI)
        self.assertEqual(
            out.strip(), "abi: no LLVM IR to compile (2 configs emit something else)"
        )
        self.assertNotRegex(
            out, r"^[A-Za-z0-9_]+: 0 compiled"
        )  # the check script's "nothing compiled" rule

    def test_refusals_develop_shares_leave_nothing_to_compile(self):
        """ir_serialize's serialized outputs and refusals, rejected on develop too, do not fail G7 as "nothing
        compiled"."""
        (self.base / "dev" / PDIR / "serial_emit.py").write_text(SERIAL)
        out = self.tool("serial", SERIAL)
        self.assertEqual(
            out.strip(),
            "serial: no LLVM IR to compile (1 configs emit something else; 1 rejected on "
            "develop too)",
        )
        out = self.tool(
            "serial2", SERIAL
        )  # a family the PR adds: its refusal has no develop baseline
        self.assertTrue(out.startswith("serial2: 0 compiled"), out)

    def test_a_config_that_fails_to_compile_on_develop_too_is_not_the_prs(self):
        """gfx942 3D cfg 2 needs more LDS than gfx942 has, on develop too: it fails G7 for no PR touching that
        family."""
        (self.base / "dev" / PDIR / "oldfail_emit.py").write_text(FAILS)
        out = self.tool("oldfail", FAILS)
        self.assertTrue(
            out.startswith(
                "oldfail: 0 compiled, 0 compile failures, 1 also fail to compile on develop"
            ),
            out,
        )
        self.assertIn("fails on develop too", out)
        (self.base / "dev" / PDIR / "newfail_emit.py").write_text(
            COMPILES
        )  # develop compiles it: the PR broke it
        out = self.tool("newfail", FAILS)
        self.assertTrue(
            out.startswith("newfail: 0 compiled, 1 compile failures, 0 newly rejected"),
            out,
        )

    def test_a_family_the_pr_adds_has_no_develop_baseline(self):
        """A family the PR adds has no develop baseline: its must-reject configs are not "rejected on develop too"."""
        out = self.tool("newfam", NEWFAM)
        self.assertTrue(
            out.startswith(
                "newfam: 1 compiled, 0 compile failures, 0 newly rejected, 0 rejected on develop "
                "too, 1 rejected (no develop baseline)"
            ),
            out,
        )

    def test_spills_name_their_configs_and_what_develop_shares(self):
        """The checks name the configs that spill and whether develop spills them too, not only "spills 7"."""
        for name, spill in (("pr", (1, 2)), ("dev", (1,))):
            (self.base / name / PDIR / "sp_emit.py").write_text(
                SPILLS.format(spill=spill)
            )
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--root",
                str(self.base / "pr"),
                "--base-root",
                str(self.base / "dev"),
                f"{PDIR}:sp",
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(
            "spills 2 (cfg 1, 2; 1 also spill on develop, new: cfg 2)", p.stdout
        )

    def test_kernels_compile_with_rockes_own_codegen_options(self):
        """AGPR bodies compile with rocKE's own -mllvm -amdgpu-mfma-vgpr-form=0, never plain -O3, which reads them as
        false spills."""
        root = self.base / "pr"
        (root / PDIR / "agpr_emit.py").write_text(AGPR)
        py = root / "platform/python/rocke"
        (py / "core").mkdir(parents=True)
        (py / "helpers").mkdir()
        for f in ("__init__.py", "core/__init__.py", "helpers/__init__.py"):
            (py / f).write_text("")
        (py / "core/lower_llvm.py").write_text(LOWER_AGPR)
        (py / "helpers/compile.py").write_text(OPTIONS)
        run = lambda: subprocess.run(
            [sys.executable, "-B", str(TOOL), "--root", str(root), f"{PDIR}:agpr"],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        p = run()
        self.assertTrue(
            p.stdout.startswith("agpr: 2 compiled, 0 compile failures"),
            p.stdout + p.stderr,
        )
        self.assertIn("spills 0", p.stdout)
        # A tree without the helper compiles plain -O3, as before: the AGPR body spills.
        (py / "helpers/compile.py").unlink()
        p = run()
        self.assertIn("spills 1 (cfg 1)", p.stdout, p.stderr)

    def test_the_config_cap_is_reported_not_silent(self):
        """The config cap is reported: configs past it are never skipped silently."""
        self.env["RKMON_MAX_CONFIGS"] = "3"
        out = self.tool("five", FIVE)
        self.assertTrue(out.startswith("five: 3 compiled"), out)
        self.assertIn(
            "stopped at the 3-config cap: later configs were not checked", out
        )
        self.env["RKMON_MAX_CONFIGS"] = "6"
        out = self.tool("five", FIVE)
        self.assertTrue(out.startswith("five: 5 compiled"), out)
        self.assertNotIn("cap", out)


if __name__ == "__main__":
    unittest.main()

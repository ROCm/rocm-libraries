# Phase-1 step 1: capture the CPython import closure of the provider's compile
# path (elementwise smoke + conv instance import), then categorize the non-ck
# modules so we know the MicroPython shim surface.
#
# Run under CPython:  LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib python3 spike/phase1_closure.py
import sys

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython"
sys.path.insert(0, ROOT + "/projects/composablekernel/python")
sys.path.insert(0, ROOT + "/dnn-providers/ck-dsl-provider/python")

base = set(sys.modules)  # interpreter-startup modules — exclude these

import ck_dsl_provider.compile_service as cs

# Full elementwise compile path (build -> compile_kernel -> comgr -> HSACO).
art = cs.compile_smoke("gfx1151")
assert art["hsaco"][:4] == b"\x7fELF", "smoke HSACO not ELF"

# Pull in the conv instance's import graph (module-level imports: dataclasses,
# typing, core.arch, IR, etc.) — the real target slice.
import ck_dsl.instances.common.conv_implicit_gemm as conv  # noqa: F401

added = sorted(m for m in sys.modules if m not in base)

ck = [m for m in added if m.split(".")[0] in ("ck_dsl", "ck_dsl_provider")]
other = [m for m in added if m.split(".")[0] not in ("ck_dsl", "ck_dsl_provider")]
# top-level non-ck packages only (dedupe submodules)
tops = sorted({m.split(".")[0] for m in other})

print("=== smoke compile OK: HSACO bytes:", len(art["hsaco"]), "isa:", art["isa"])
print("=== ck_dsl* / ck_dsl_provider* modules:", len(ck))
print("=== non-ck modules:", len(other), " top-level packages:", len(tops))
print("--- non-ck top-level packages ---")
print(" ".join(tops))
print("--- full non-ck module list ---")
print(" ".join(other))

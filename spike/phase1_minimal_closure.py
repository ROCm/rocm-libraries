# Phase-1 step 2: measure the TRUE minimal import closure of the compile path,
# bypassing the eager-import bloat in ck_dsl/__init__.py, helpers/__init__.py,
# instances/__init__.py, runtime/__init__.py.
#
# Trick: pre-seed sys.modules with namespace stubs for the heavy *packages*
# (real __path__, empty body) so Python imports their LEAF modules without
# running the heavy __init__.py. Non-destructive (no file edits).
#
# Run: LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib python3 spike/phase1_minimal_closure.py
import sys
import types

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython"
PKG = ROOT + "/projects/composablekernel/python/ck_dsl"
sys.path.insert(0, ROOT + "/projects/composablekernel/python")


def stub(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


# Neuter the heavy package __init__ bodies (keep submodule importability).
stub("ck_dsl", PKG)
stub("ck_dsl.helpers", PKG + "/helpers")
stub("ck_dsl.instances", PKG + "/instances")
stub("ck_dsl.runtime", PKG + "/runtime")

base = set(sys.modules)  # excludes the stubs above


def report(tag):
    added = sorted(m for m in sys.modules if m not in base)
    other = [m for m in added if m.split(".")[0] not in ("ck_dsl", "ck_dsl_provider")]
    ck = [m for m in added if m.split(".")[0] in ("ck_dsl", "ck_dsl_provider")]
    tops = sorted({m.split(".")[0] for m in other})
    print(
        "=== [%s] ck modules: %d  non-ck: %d  non-ck top-level: %d"
        % (tag, len(ck), len(other), len(tops))
    )
    print("    non-ck top-level:", " ".join(tops))


# (1) Elementwise smoke — the clean minimal slice.
from ck_dsl.helpers.compile import compile_kernel
from ck_dsl.instances.common.elementwise import ElementwiseSpec, build_elementwise

spec = ElementwiseSpec(
    op="copy", dtype="f16", block_size=64, vec=2, name="mp_min_smoke"
)
art = compile_kernel(build_elementwise(spec), arch="gfx1151")
assert art.hsaco[:4] == b"\x7fELF"
print("=== elementwise minimal compile OK, HSACO:", len(art.hsaco), "isa:", art.isa)
report("elementwise")

# (2) Conv — does it stay minimal, or pull analysis/helpers coupling?
#     Add stubs for analysis/benchmark too and see how far we get.
stub("ck_dsl.analysis", PKG + "/analysis")
stub("ck_dsl.benchmark", PKG + "/benchmark")
try:
    import ck_dsl.instances.common.conv_implicit_gemm  # noqa: F401

    print("=== conv import OK with analysis/benchmark stubbed")
    report("elementwise+conv")
except Exception as e:
    print("=== conv import FAILED even with analysis/benchmark stubbed:")
    print("   ", type(e).__name__, e)

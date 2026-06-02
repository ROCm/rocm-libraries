# Phase-1 step 3: for the minimal elementwise compile path, record WHO first
# imports each non-ck top-level package, so we can tell lazy-removable imports
# (e.g. subprocess for the unused hipcc fallback) from genuinely-needed ones.
import builtins
import sys
import types

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython"
PKG = ROOT + "/projects/composablekernel/python/ck_dsl"
sys.path.insert(0, ROOT + "/projects/composablekernel/python")


def stub(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


for n, p in [
    ("ck_dsl", PKG),
    ("ck_dsl.helpers", PKG + "/helpers"),
    ("ck_dsl.instances", PKG + "/instances"),
    ("ck_dsl.runtime", PKG + "/runtime"),
]:
    stub(n, p)

SUSPECTS = {
    "subprocess",
    "tempfile",
    "inspect",
    "ast",
    "dis",
    "tokenize",
    "token",
    "bz2",
    "lzma",
    "urllib",
    "ipaddress",
    "shutil",
    "glob",
    "locale",
    "decimal",
    "fractions",
    "statistics",
    "weakref",
    "ctypes",
    "enum",
    "dataclasses",
    "typing",
    "pathlib",
    "json",
    "re",
}

_real_import = builtins.__import__
first_by = {}


def _traced(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top in SUSPECTS and top not in first_by and top not in sys.modules:
        first_by[top] = (globals or {}).get("__name__", "<top>")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _traced

from ck_dsl.helpers.compile import compile_kernel
from ck_dsl.instances.common.elementwise import ElementwiseSpec, build_elementwise

spec = ElementwiseSpec(op="copy", dtype="f16", block_size=64, vec=2, name="mp_trace")
art = compile_kernel(build_elementwise(spec), arch="gfx1151")
builtins.__import__ = _real_import
assert art.hsaco[:4] == b"\x7fELF"

print("=== who first imports each suspect (top-level pkg <- importing module) ===")
for k in sorted(first_by):
    print("  %-14s <- %s" % (k, first_by[k]))
missing = sorted(s for s in SUSPECTS if s not in first_by and s not in sys.modules)

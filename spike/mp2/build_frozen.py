#!/usr/bin/env python3
# Build spike/mp2/frozen_src/: the exact conv codegen closure (ck modules + shims)
# to freeze into the embed binary. Runs under CPython as a build step.
#   - captures the closure by importing the conv codegen path against the mp1 bundle
#   - copies those modules + ALL shims (embed port lacks functools/itertools/etc.)
#   - embeds arch_specs.json as a Python module (frozen modules have no filesystem)
#   - patches arch/target.py off __file__/open/json onto the embedded dict
import json
import os
import shutil
import sys

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython"
MP1 = ROOT + "/spike/mp1"
BUNDLE = MP1 + "/ckbundle"
SHIMS = MP1 + "/shims"
FROZEN = ROOT + "/spike/mp2/frozen_src"

# 1. Capture the conv codegen closure.
sys.path.insert(0, BUNDLE)
sys.path.insert(0, SHIMS)
from ck_dsl.instances.common.conv_implicit_gemm import (  # noqa: E402
    ImplicitGemmConvSpec,
    ConvProblem,
    build_implicit_gemm_conv,
)
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402

spec = ImplicitGemmConvSpec(
    problem=ConvProblem(
        N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
    ),
    tile_m=64,
    tile_n=64,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_tile_m=32,
    warp_tile_n=32,
    warp_tile_k=16,
)
_ = lower_kernel_to_llvm(build_implicit_gemm_conv(spec, arch="gfx950"), arch="gfx950")

ck_files = sorted(
    m.__file__[len(BUNDLE) + 1 :]
    for m in sys.modules.values()
    if getattr(m, "__file__", None) and m.__file__.startswith(BUNDLE + "/")
)

# 2. Reset frozen tree; copy ck closure + ALL shims.
if os.path.exists(FROZEN):
    shutil.rmtree(FROZEN)
os.makedirs(FROZEN)
for rel in ck_files:
    dst = os.path.join(FROZEN, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(BUNDLE, rel), dst)
for shim in os.listdir(SHIMS):
    if shim.endswith(".py"):
        shutil.copy(os.path.join(SHIMS, shim), os.path.join(FROZEN, shim))

# 3. Embed arch_specs.json as a Python module.
with open(os.path.join(BUNDLE, "ck_dsl/core/arch/data/arch_specs.json")) as f:
    doc = json.load(f)
with open(os.path.join(FROZEN, "ck_dsl/core/arch/_arch_specs_embedded.py"), "w") as f:
    f.write(
        "# Generated: arch_specs.json embedded for the frozen (no-filesystem) build.\n"
    )
    f.write("DOC = " + repr(doc) + "\n")

# 4. Patch target.py off __file__/open/json onto the embedded dict.
tp = os.path.join(FROZEN, "ck_dsl/core/arch/target.py")
s = open(tp).read()
s = s.replace(
    "_DATA_FILE = Path(__file__).parent / 'data' / 'arch_specs.json'",
    "from ._arch_specs_embedded import DOC as _EMBEDDED_DOC\n"
    "_DATA_FILE = Path('arch_specs.json')  # name only; data is embedded",
)
s = s.replace(
    "    with open(str(_DATA_FILE)) as fh:\n        doc = json.load(fh)\n",
    "    doc = _EMBEDDED_DOC\n",
)
s = s.replace("import json\n", "")  # json.load patched out; embed has no json module
open(tp, "w").write(s)
assert "_EMBEDDED_DOC" in open(tp).read(), "target.py patch failed"

# 5. Entry module the C host imports + calls. Mirrors ck_dsl's flow: lower to IR,
#    then call comgr (exposed as a native module) -> returns the HSACO bytes, like
#    runtime/comgr.py would. So the C host gets a HSACO, not IR text.
with open(os.path.join(FROZEN, "ckdsl_entry.py"), "w") as f:
    f.write(
        "from ck_dsl.instances.common.conv_implicit_gemm import (\n"
        "    ImplicitGemmConvSpec, ConvProblem, build_implicit_gemm_conv)\n"
        "from ck_dsl.core.lower_llvm import lower_kernel_to_llvm\n\n\n"
        "def compile_conv():\n"
        "    spec = ImplicitGemmConvSpec(\n"
        "        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3,\n"
        "                            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),\n"
        "        tile_m=64, tile_n=64, tile_k=64, warp_m=2, warp_n=2,\n"
        "        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)\n"
        "    ir = lower_kernel_to_llvm(\n"
        "        build_implicit_gemm_conv(spec, arch='gfx950'), arch='gfx950')\n"
        "    import comgr  # native C++ module exposed to the interpreter\n"
        "    return comgr.build_hsaco(ir, 'amdgcn-amd-amdhsa--gfx950', ['-O3'])\n"
    )

n = sum(len(fs) for _, _, fs in os.walk(FROZEN))
print(
    "frozen_src: %d ck modules + shims, %d files total at %s"
    % (len(ck_files), n, FROZEN)
)

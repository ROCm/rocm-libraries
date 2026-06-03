# Capture the exact module files the conv codegen path loads (run under CPython
# against the mp1 bundle + shims), so we can freeze precisely that set.
import sys

MP1 = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1"
sys.path.insert(0, MP1 + "/ckbundle")
sys.path.insert(0, MP1 + "/shims")

from ck_dsl.instances.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    ConvProblem,
    build_implicit_gemm_conv,
)
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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

ck, shim = [], []
for name, mod in sys.modules.items():
    f = getattr(mod, "__file__", None)
    if not f:
        continue
    if f.startswith(MP1 + "/ckbundle/"):
        ck.append(f[len(MP1 + "/ckbundle/") :])
    elif f.startswith(MP1 + "/shims/"):
        shim.append(f[len(MP1 + "/shims/") :])
print("CK_COUNT", len(ck), "SHIM_COUNT", len(shim))
print("=== shims ===")
[print(s) for s in sorted(shim)]
print("=== ck (first 50) ===")
[print(c) for c in sorted(ck)[:50]]
import json

open("/tmp/closure.json", "w").write(
    json.dumps({"ck": sorted(ck), "shim": sorted(shim)})
)

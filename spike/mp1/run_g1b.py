# G1b: conv codegen -> LLVM IR -> comgr-ffi -> HSACO, entirely in MicroPython.
import sys

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1"
sys.path.insert(0, ROOT + "/ckbundle")
sys.path.insert(0, ROOT + "/shims")
sys.path.insert(0, ROOT)  # comgr_ffi

from ck_dsl.instances.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    ConvProblem,
    build_implicit_gemm_conv,
)
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm
import comgr_ffi

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
kdef = build_implicit_gemm_conv(spec, arch="gfx950")
llvm = lower_kernel_to_llvm(kdef, arch="gfx950")
print("LLVM_LEN", len(llvm))

opts = ["-O3"]
if kdef.attrs.get("mfma_vgpr_form"):
    opts += ["-mllvm", "-amdgpu-mfma-vgpr-form"]
hsaco = comgr_ffi.build_hsaco_from_llvm_ir(
    llvm, isa="amdgcn-amd-amdhsa--gfx950", options=opts
)
print("HSACO_LEN", len(hsaco))
print("ELF?", hsaco[:4] == b"\x7fELF")
with open("/tmp/conv_mp.hsaco", "wb") as f:
    f.write(hsaco)
print("G1b PASS: conv codegen + comgr -> HSACO, all in MicroPython")

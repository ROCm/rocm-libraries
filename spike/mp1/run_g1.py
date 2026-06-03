import sys

ROOT = "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1"
sys.path.insert(0, ROOT + "/ckbundle")
sys.path.insert(
    0, ROOT + "/shims"
)  # shims win for dataclasses/typing/functools/pathlib

from ck_dsl.instances.common.elementwise import ElementwiseSpec, build_elementwise
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

spec = ElementwiseSpec(op="copy", dtype="f16", block_size=64, vec=2, name="mp_g1")
kdef = build_elementwise(spec)
llvm = lower_kernel_to_llvm(kdef, arch="gfx1151")
print("LLVM_LEN", len(llvm))
print("LLVM_SUM", sum(bytearray(llvm.encode("utf-8"))))
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/g1.ll"
with open(out, "w") as f:
    f.write(llvm)

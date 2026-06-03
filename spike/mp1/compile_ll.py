# Compile a .ll file to HSACO via comgr-ffi (MicroPython). Usage: compile_ll.py in.ll out.hsaco
import sys

sys.path.insert(
    0, "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1"
)
import comgr_ffi

with open(sys.argv[1]) as f:
    ir = f.read()
hsaco = comgr_ffi.build_hsaco_from_llvm_ir(
    ir, isa="amdgcn-amd-amdhsa--gfx950", options=["-O3"]
)
with open(sys.argv[2], "wb") as f:
    f.write(hsaco)
print(sys.argv[1], "-> HSACO", len(hsaco))

# Phase-0 spike: drive libamd_comgr's 3-stage compile from MicroPython via ffi/uctypes.
# Mirrors projects/composablekernel/python/ck_dsl/runtime/comgr.py, but with NO ctypes:
#   - opaque handles (single-uint64 structs by value) are passed as "Q" (SysV ABI equivalence)
#   - out-params / buffers / char** arrays are bytearrays addressed via uctypes
# Success = a valid ELF/HSACO blob comes out the far end.

import ffi
import uctypes

COMGR = "/opt/rocm-7.2.4/lib/libamd_comgr.so"

OK = 0
KIND_SOURCE = 0x1
KIND_EXECUTABLE = 0x8
LANG_LLVM_IR = 0x4
ACT_COMPILE_SOURCE_TO_BC = 0x2
ACT_CODEGEN_BC_TO_RELOCATABLE = 0x4
ACT_LINK_RELOCATABLE_TO_EXECUTABLE = 0x7

lib = ffi.open(COMGR)

# Bindings. ret "i" = amd_comgr_status_t. Handles by value -> "Q". Pointers/buffers -> "p".
create_data_set = lib.func("i", "amd_comgr_create_data_set", "p")  # out _DataSet*
destroy_data_set = lib.func("i", "amd_comgr_destroy_data_set", "Q")
create_data = lib.func("i", "amd_comgr_create_data", "ip")  # kind, out _Data*
release_data = lib.func("i", "amd_comgr_release_data", "Q")
set_data = lib.func("i", "amd_comgr_set_data", "QQp")  # data, size, bytes
set_data_name = lib.func("i", "amd_comgr_set_data_name", "Qs")  # data, char*
data_set_add = lib.func("i", "amd_comgr_data_set_add", "QQ")  # set, data
action_data_count = lib.func(
    "i", "amd_comgr_action_data_count", "Qip"
)  # set, kind, out size_t*
action_data_get_data = lib.func(
    "i", "amd_comgr_action_data_get_data", "QiQp"
)  # set, kind, idx, out _Data*
get_data = lib.func(
    "i", "amd_comgr_get_data", "Qpp"
)  # data, in/out size_t*, char*|NULL
create_action_info = lib.func("i", "amd_comgr_create_action_info", "p")
destroy_action_info = lib.func("i", "amd_comgr_destroy_action_info", "Q")
set_isa = lib.func("i", "amd_comgr_action_info_set_isa_name", "Qs")
set_lang = lib.func("i", "amd_comgr_action_info_set_language", "Qi")
set_options = lib.func(
    "i", "amd_comgr_action_info_set_option_list", "QpQ"
)  # info, char**, count
do_action = lib.func("i", "amd_comgr_do_action", "iQQQ")  # kind, info, in_set, out_set


def u64(ba):
    return int.from_bytes(ba, "little")


def check(s, where):
    if s != OK:
        raise RuntimeError("%s: comgr status=%d" % (where, s))


def new_handle(fn, where, *pre):
    out = bytearray(8)
    check(fn(*(pre + (out,))), where)
    return u64(out)


def extract_executable(out_set):
    cnt = bytearray(8)
    check(action_data_count(out_set, KIND_EXECUTABLE, cnt), "action_data_count")
    if u64(cnt) == 0:
        raise RuntimeError("no EXECUTABLE produced")
    dh = bytearray(8)
    check(action_data_get_data(out_set, KIND_EXECUTABLE, 0, dh), "action_data_get_data")
    data = u64(dh)
    size = bytearray(8)
    check(get_data(data, size, 0), "get_data(size)")  # NULL buffer -> query size
    n = u64(size)
    buf = bytearray(n)
    check(get_data(data, size, buf), "get_data(read)")
    release_data(data)
    return bytes(buf[:n])


def build_hsaco(ir_text, isa, options):
    # Input data set: LLVM IR wrapped as SOURCE.
    in_set = new_handle(create_data_set, "create_data_set(in)")
    src = new_handle(create_data, "create_data(src)", KIND_SOURCE)
    payload = ir_text.encode("utf-8")
    check(set_data(src, len(payload), payload), "set_data")
    check(set_data_name(src, b"kernel.ll"), "set_data_name")
    check(data_set_add(in_set, src), "data_set_add")

    # Action info.
    info = new_handle(create_action_info, "create_action_info")
    check(set_isa(info, isa.encode("utf-8")), "set_isa")
    check(set_lang(info, LANG_LLVM_IR), "set_lang")
    opt_bytes = [o.encode("utf-8") for o in options]  # keep alive
    arr = bytearray(8 * len(opt_bytes))
    for i, o in enumerate(opt_bytes):
        arr[i * 8 : i * 8 + 8] = uctypes.addressof(o).to_bytes(8, "little")
    check(set_options(info, arr, len(opt_bytes)), "set_options")

    # Stage 1: SOURCE -> BC
    bc_set = new_handle(create_data_set, "create_data_set(bc)")
    check(
        do_action(ACT_COMPILE_SOURCE_TO_BC, info, in_set, bc_set),
        "do_action(SOURCE_TO_BC)",
    )
    # Stage 2: BC -> RELOCATABLE
    reloc_set = new_handle(create_data_set, "create_data_set(reloc)")
    check(
        do_action(ACT_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set),
        "do_action(BC_TO_RELOCATABLE)",
    )
    # Stage 3: RELOCATABLE -> EXECUTABLE
    exe_set = new_handle(create_data_set, "create_data_set(exe)")
    check(
        do_action(ACT_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set, exe_set),
        "do_action(RELOC_TO_EXE)",
    )

    hsaco = extract_executable(exe_set)

    release_data(src)
    for h in (in_set, bc_set, reloc_set, exe_set):
        destroy_data_set(h)
    destroy_action_info(info)
    return hsaco


IR = """\
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @empty() #0 {
entry:
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
"""

isa = "amdgcn-amd-amdhsa--gfx1151"
hsaco = build_hsaco(IR, isa, ["-O3"])
magic = hsaco[:4]
print("HSACO bytes:", len(hsaco))
print("magic:", magic, "ELF?", magic == b"\x7fELF")
with open(
    "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/out.hsaco", "wb"
) as f:
    f.write(hsaco)
print("G0 PASS: comgr 3-stage compile driven entirely via MicroPython ffi/uctypes")

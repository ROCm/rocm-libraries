# comgr via MicroPython ffi/uctypes — module form of the Phase-0 spike.
# Drives libamd_comgr's SOURCE->BC->RELOCATABLE->EXECUTABLE chain with no ctypes.
# This is the `runtime/comgr` replacement for a MicroPython ck_dsl bundle.
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

_lib = ffi.open(COMGR)
_create_data_set = _lib.func("i", "amd_comgr_create_data_set", "p")
_destroy_data_set = _lib.func("i", "amd_comgr_destroy_data_set", "Q")
_create_data = _lib.func("i", "amd_comgr_create_data", "ip")
_release_data = _lib.func("i", "amd_comgr_release_data", "Q")
_set_data = _lib.func("i", "amd_comgr_set_data", "QQp")
_set_data_name = _lib.func("i", "amd_comgr_set_data_name", "Qs")
_data_set_add = _lib.func("i", "amd_comgr_data_set_add", "QQ")
_action_data_count = _lib.func("i", "amd_comgr_action_data_count", "Qip")
_action_data_get_data = _lib.func("i", "amd_comgr_action_data_get_data", "QiQp")
_get_data = _lib.func("i", "amd_comgr_get_data", "Qpp")
_create_action_info = _lib.func("i", "amd_comgr_create_action_info", "p")
_destroy_action_info = _lib.func("i", "amd_comgr_destroy_action_info", "Q")
_set_isa = _lib.func("i", "amd_comgr_action_info_set_isa_name", "Qs")
_set_lang = _lib.func("i", "amd_comgr_action_info_set_language", "Qi")
_set_options = _lib.func("i", "amd_comgr_action_info_set_option_list", "QpQ")
_do_action = _lib.func("i", "amd_comgr_do_action", "iQQQ")


class ComgrError(RuntimeError):
    pass


def _u64(ba):
    return int.from_bytes(ba, "little")


def _check(s, where):
    if s != OK:
        raise ComgrError("%s: comgr status=%d" % (where, s))


def _new(fn, where, *pre):
    out = bytearray(8)
    _check(fn(*(pre + (out,))), where)
    return _u64(out)


def _extract_executable(out_set):
    cnt = bytearray(8)
    _check(_action_data_count(out_set, KIND_EXECUTABLE, cnt), "action_data_count")
    if _u64(cnt) == 0:
        raise ComgrError("no EXECUTABLE produced")
    dh = bytearray(8)
    _check(
        _action_data_get_data(out_set, KIND_EXECUTABLE, 0, dh), "action_data_get_data"
    )
    data = _u64(dh)
    size = bytearray(8)
    _check(_get_data(data, size, 0), "get_data(size)")
    n = _u64(size)
    buf = bytearray(n)
    _check(_get_data(data, size, buf), "get_data(read)")
    _release_data(data)
    return bytes(buf[:n])


def build_hsaco_from_llvm_ir(ir_text, isa="amdgcn-amd-amdhsa--gfx950", options=None):
    options = list(options or ["-O3"])
    in_set = _new(_create_data_set, "create_data_set(in)")
    src = _new(_create_data, "create_data(src)", KIND_SOURCE)
    payload = ir_text.encode("utf-8")
    _check(_set_data(src, len(payload), payload), "set_data")
    _check(_set_data_name(src, b"kernel.ll"), "set_data_name")
    _check(_data_set_add(in_set, src), "data_set_add")

    info = _new(_create_action_info, "create_action_info")
    _check(_set_isa(info, isa.encode("utf-8")), "set_isa")
    _check(_set_lang(info, LANG_LLVM_IR), "set_lang")
    opt_bytes = [o.encode("utf-8") for o in options]
    arr = bytearray(8 * len(opt_bytes))
    for i, o in enumerate(opt_bytes):
        arr[i * 8 : i * 8 + 8] = uctypes.addressof(o).to_bytes(8, "little")
    _check(_set_options(info, arr, len(opt_bytes)), "set_options")

    bc = _new(_create_data_set, "create_data_set(bc)")
    _check(_do_action(ACT_COMPILE_SOURCE_TO_BC, info, in_set, bc), "do_action(BC)")
    reloc = _new(_create_data_set, "create_data_set(reloc)")
    _check(
        _do_action(ACT_CODEGEN_BC_TO_RELOCATABLE, info, bc, reloc), "do_action(RELOC)"
    )
    exe = _new(_create_data_set, "create_data_set(exe)")
    _check(
        _do_action(ACT_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc, exe),
        "do_action(EXE)",
    )

    hsaco = _extract_executable(exe)
    _release_data(src)
    for h in (in_set, bc, reloc, exe):
        _destroy_data_set(h)
    _destroy_action_info(info)
    return hsaco

/* Native MicroPython module exposing comgr to the interpreter: comgr is in C++
 * (links amd_comgr, no MicroPython ffi -> Windows-friendly), but callable from
 * Python so ck_dsl keeps its existing flow (comgr.py backend -> this; compile_service
 * still returns the HSACO dict). Wraps comgr_compile.c.
 *
 *   comgr.build_hsaco(ir: str|bytes, isa: str, options: list[str]) -> bytes
 */
#include <stdlib.h>

#include "comgr_compile.h"
#include "py/objstr.h"
#include "py/runtime.h"

static mp_obj_t mod_comgr_build_hsaco(mp_obj_t ir_in, mp_obj_t isa_in, mp_obj_t opts_in) {
    size_t ir_len;
    const char* ir = mp_obj_str_get_data(ir_in, &ir_len);
    const char* isa = mp_obj_str_get_str(isa_in);

    size_t n;
    mp_obj_t* items;
    mp_obj_get_array(opts_in, &n, &items);
    // GC-managed array: if mp_obj_str_get_str() raises on a non-str option mid
    // loop (longjmp), the GC reclaims it -- a malloc'd array would leak.
    const char** opts = m_new(const char*, n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        opts[i] = mp_obj_str_get_str(items[i]);
    }

    unsigned char* out = NULL;
    size_t out_len = 0;
    char err[256];
    err[0] = '\0';
    int rc = comgr_build_hsaco(ir, ir_len, isa, opts, n, &out, &out_len, err, sizeof(err));
    if (rc != 0) {
        if (out != NULL) {
            free(out);
        }
        mp_raise_msg_varg(&mp_type_RuntimeError, MP_ERROR_TEXT("comgr build_hsaco failed: %s"),
                          err[0] != '\0' ? err : "unknown");
    }

    // Copy the comgr buffer into a MicroPython bytes object, freeing `out` on
    // every path -- mp_obj_new_bytes can raise (OOM) before we reach the free.
    nlr_buf_t nlr;
    mp_obj_t res;
    if (nlr_push(&nlr) == 0) {
        res = mp_obj_new_bytes(out, out_len);
        nlr_pop();
    } else {
        free(out);
        nlr_jump(nlr.ret_val);
    }
    free(out);
    return res;
}
static MP_DEFINE_CONST_FUN_OBJ_3(mod_comgr_build_hsaco_obj, mod_comgr_build_hsaco);

static const mp_rom_map_elem_t comgr_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_comgr)},
    {MP_ROM_QSTR(MP_QSTR_build_hsaco), MP_ROM_PTR(&mod_comgr_build_hsaco_obj)},
};
static MP_DEFINE_CONST_DICT(comgr_module_globals, comgr_module_globals_table);

const mp_obj_module_t comgr_user_cmodule = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t*)&comgr_module_globals,
};
MP_REGISTER_MODULE(MP_QSTR_comgr, comgr_user_cmodule);

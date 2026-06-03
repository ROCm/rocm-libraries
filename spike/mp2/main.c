/* Arch-A integration harness (comgr-as-native-module variant): embed MicroPython,
 * call the frozen ck_dsl flow from C; ck_dsl lowers to IR and calls the native
 * `comgr` module (C++ libamd_comgr, no ffi) which returns the HSACO bytes -- so the
 * C host receives a HSACO, exactly like ck_dsl's existing compile flow. */
#include <stdio.h>
#include <stdlib.h>

#include "port/micropython_embed.h"
#include "py/mpprint.h"
#include "py/objstr.h"
#include "py/runtime.h"

int main(void) {
    int stack_top;
    size_t heap_size = (size_t)256 * 1024 * 1024;
    char* heap = malloc(heap_size);
    if (!heap) {
        fprintf(stderr, "heap alloc failed\n");
        return 1;
    }
    mp_embed_init(heap, heap_size, &stack_top);

    int rc = 0;
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        // import ckdsl_entry; hsaco = ckdsl_entry.compile_conv()  (returns HSACO bytes)
        mp_obj_t mod =
            mp_import_name(qstr_from_str("ckdsl_entry"), mp_const_none, MP_OBJ_NEW_SMALL_INT(0));
        mp_obj_t func = mp_load_attr(mod, qstr_from_str("compile_conv"));
        mp_obj_t hsaco = mp_call_function_0(func);

        size_t len;
        const char* s = mp_obj_str_get_data(hsaco, &len);  // works for bytes
        int is_elf =
            len >= 4 && (unsigned char)s[0] == 0x7f && s[1] == 'E' && s[2] == 'L' && s[3] == 'F';
        unsigned long sum = 0;
        for (size_t i = 0; i < len; i++) {
            sum += (unsigned char)s[i];
        }
        printf("C received HSACO from Python (native comgr): %u bytes, ELF=%d, sum=%lu\n",
               (unsigned)len, is_elf, sum);
        if (is_elf) {
            FILE* f = fopen("/tmp/embed_conv.hsaco", "wb");
            if (f) {
                fwrite(s, 1, len, f);
                fclose(f);
            }
        } else {
            rc = 1;
        }
        nlr_pop();
    } else {
        rc = 1;
        printf("Python exception during compile_conv:\n");
        mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(nlr.ret_val));
    }

    mp_embed_deinit();
    free(heap);
    return rc;
}

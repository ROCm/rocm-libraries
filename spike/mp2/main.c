/* Arch-A integration harness: embed MicroPython, call the frozen ck_dsl codegen
 * from C, and capture the LLVM IR string into C (no filesystem, no system Python).
 * Next step grafts a C++ comgr wrapper (IR -> HSACO) onto the captured string. */
#include <stdio.h>
#include <stdlib.h>

#include "comgr_compile.h"
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
        // import ckdsl_entry; ir = ckdsl_entry.compile_conv()
        mp_obj_t mod =
            mp_import_name(qstr_from_str("ckdsl_entry"), mp_const_none, MP_OBJ_NEW_SMALL_INT(0));
        mp_obj_t func = mp_load_attr(mod, qstr_from_str("compile_conv"));
        mp_obj_t ir = mp_call_function_0(func);

        size_t len;
        const char* s = mp_obj_str_get_data(ir, &len);
        unsigned long sum = 0;
        for (size_t i = 0; i < len; i++) {
            sum += (unsigned char)s[i];
        }
        printf("C captured IR: len=%u sum=%lu\n", (unsigned)len, sum);

        // Arch A: C++ comgr turns the captured IR into a HSACO (GPU code object).
        const char* opts[] = {"-O3"};
        unsigned char* hsaco = NULL;
        size_t hlen = 0;
        int crc = comgr_build_hsaco(s, len, "amdgcn-amd-amdhsa--gfx950", opts, 1, &hsaco, &hlen);
        if (crc == 0 && hlen >= 4 && hsaco[0] == 0x7f && hsaco[1] == 'E' && hsaco[2] == 'L' &&
            hsaco[3] == 'F') {
            printf("HSACO: %u bytes, valid ELF (DSL->IR->HSACO all in one process)\n",
                   (unsigned)hlen);
            FILE* f = fopen("/tmp/embed_conv.hsaco", "wb");
            if (f) {
                fwrite(hsaco, 1, hlen, f);
                fclose(f);
            }
        } else {
            printf("comgr failed: rc=%d hlen=%u\n", crc, (unsigned)hlen);
            rc = 1;
        }
        free(hsaco);
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

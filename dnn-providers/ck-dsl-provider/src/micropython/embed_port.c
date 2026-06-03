/* Port-provided stubs for the embed build with external import enabled.
 * We have no filesystem — modules are frozen — so filesystem stat always
 * reports "doesn't exist" (the ".frozen/" prefix is handled in builtinimport.c
 * before mp_import_stat), and source-file loading is never reached. */
#include "py/builtin.h"
#include "py/lexer.h"
#include "py/runtime.h"

mp_import_stat_t mp_import_stat(const char* path) {
    (void)path;
    return MP_IMPORT_STAT_NO_EXIST;
}

mp_lexer_t* mp_lexer_new_from_file(qstr filename) {
    (void)filename;
    mp_raise_NotImplementedError(MP_ERROR_TEXT("filesystem source import not supported"));
}

#include "py/mperrno.h"

// No filesystem: open() exists (so ck_dsl's `try: open(...) except OSError` paths
// fall back cleanly) but always fails.
static mp_obj_t embed_builtin_open(size_t n_args, const mp_obj_t* args, mp_map_t* kwargs) {
    (void)n_args;
    (void)args;
    (void)kwargs;
    mp_raise_OSError(MP_ENOENT);
}
MP_DEFINE_CONST_FUN_OBJ_KW(mp_builtin_open_obj, 1, embed_builtin_open);

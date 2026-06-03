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

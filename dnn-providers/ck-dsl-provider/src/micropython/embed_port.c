/* Port-provided import/filesystem hooks for the embed build.
 *
 * Frozen mode (default): there is no filesystem -- modules are frozen into the
 * binary -- so mp_import_stat always reports "doesn't exist" (the ".frozen/"
 * prefix is handled in builtinimport.c before mp_import_stat) and source-file
 * loading is never reached.
 *
 * On-disk mode (CKDSL_ON_DISK): ck_dsl/shims are loaded as .py/.mpy from the
 * filesystem. mp_reader_new_file / mp_lexer_new_from_file / mp_raw_code_load_file
 * come from py/ core (MICROPY_READER_POSIX); we only provide a real stat here.
 */
#include "py/builtin.h"
#include "py/lexer.h"
#include "py/runtime.h"

#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK

#include <sys/stat.h>

mp_import_stat_t mp_import_stat(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return MP_IMPORT_STAT_NO_EXIST;
    }
    return S_ISDIR(st.st_mode) ? MP_IMPORT_STAT_DIR : MP_IMPORT_STAT_FILE;
}

#else

mp_import_stat_t mp_import_stat(const char* path) {
    (void)path;
    return MP_IMPORT_STAT_NO_EXIST;
}

mp_lexer_t* mp_lexer_new_from_file(qstr filename) {
    (void)filename;
    mp_raise_NotImplementedError(MP_ERROR_TEXT("filesystem source import not supported"));
}

#endif

#include "py/mperrno.h"

// The Python-level open() always fails (the import path uses mp_reader_new_file
// directly, not this builtin). ck_dsl reads data files only through
// `try: open(...) except OSError` fallbacks, and its real data (arch_specs) is
// embedded, so a failing open() is correct in every mode.
static mp_obj_t embed_builtin_open(size_t n_args, const mp_obj_t* args, mp_map_t* kwargs) {
    (void)n_args;
    (void)args;
    (void)kwargs;
    mp_raise_OSError(MP_ENOENT);
}
MP_DEFINE_CONST_FUN_OBJ_KW(mp_builtin_open_obj, 1, embed_builtin_open);

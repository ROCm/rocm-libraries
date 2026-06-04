/* MicroPython embed-port config for the ck-dsl-provider Arch-A integration.
 * Starts from the minimal embed config and adds what ck_dsl codegen needs:
 * arbitrary-precision ints (MPZ), double floats, sys.modules. Frozen modules
 * carry the shims + the transformed ck_dsl codegen closure (no filesystem).
 */
#include <port/mpconfigport_common.h>

// Minimal base (disables optional extmod modules the bare embed port doesn't
// compile); we raise features incrementally as the bundle needs them.
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES)

#define MICROPY_ENABLE_COMPILER (1)
#define MICROPY_ENABLE_GC (1)
#define MICROPY_PY_GC (1)

// ck_dsl needs sys + sys.modules (lower_llvm checks sys.modules for torch).
// modsys.c references MICROPY_BANNER_MACHINE unconditionally, which needs
// MICROPY_PY_SYS_PLATFORM defined as a string (the bare embed port omits it).
#define MICROPY_PY_SYS (1)
#define MICROPY_PY_SYS_MODULES (1)
#define MICROPY_PY_SYS_PATH (1)  // frozen modules are searched via a ".frozen" sys.path entry

// Required so import searches sys.path / frozen modules (off at MINIMUM rom level).
#define MICROPY_ENABLE_EXTERNAL_IMPORT (1)
#define MICROPY_PY_SYS_PLATFORM "embed"

// Doubles for ck_dsl arithmetic. Big ints via MPZ — frozen .mpy from mpy-cross
// require MICROPY_LONGINT_IMPL == MPZ and MPZ_DIG_SIZE == 16, so match both.
#define MICROPY_FLOAT_IMPL (MICROPY_FLOAT_IMPL_DOUBLE)
#define MICROPY_LONGINT_IMPL (MICROPY_LONGINT_IMPL_MPZ)
#define MPZ_DIG_SIZE (16)  // match mpy-cross (defaults to 32 on 64-bit hosts)

// @property is used by the dataclasses/pathlib shims and ck_dsl.
#define MICROPY_PY_BUILTINS_PROPERTY (1)

// ck_dsl + the dataclasses shim need __dict__/__bases__/__name__ (off at MINIMUM).
#define MICROPY_PY_BUILTINS_FROZENSET (1)
#define MICROPY_CPYTHON_COMPAT (1)

// object.__setattr__ — ck_dsl's frozen-dataclass __init__s use it (the canonical
// CPython idiom for setting fields on a frozen instance). Enabling it lets the
// source run unmodified instead of being rewritten to setattr() at bundle time.
#define MICROPY_PY_DELATTR_SETATTR (1)

// No filesystem: keep io/open out (CORE_FEATURES would enable them).
#define MICROPY_PY_IO (1)

// Chain the frozen modules' qstr pool onto the runtime qstr pool, so frozen
// bytecode's qstr indices resolve (else find_qstr asserts). Standard freezing wiring.
#if MICROPY_MODULE_FROZEN_MPY
#define MICROPY_QSTR_EXTRA_POOL mp_qstr_frozen_const_pool
#endif

// On-disk distribution modes (CKDSL_ON_DISK): instead of frozen bytecode, load
// ck_dsl/shims as .py (or .mpy) files from the filesystem beside the plugin.
// MICROPY_READER_POSIX gives py/reader.c + py/lexer.c a real file reader/lexer;
// the embed port supplies mp_import_stat (embed_port.c). PERSISTENT_CODE_LOAD
// lets the import machinery load .mpy files (the mpy mode); harmless for py mode.
#if defined(CKDSL_ON_DISK) && CKDSL_ON_DISK
#define MICROPY_READER_POSIX (1)
#define MICROPY_PERSISTENT_CODE_LOAD (1)
// .py modules are compiled by the embedded RUNTIME compiler (not mpy-cross), so
// it must support every syntax ck_dsl uses. CORE_FEATURES leaves several at
// EXTRA level off; enable the ones ck_dsl needs (f-strings are pervasive). These
// are parser-only flags (no object-layout impact); harmless in mpy mode.
#define MICROPY_PY_FSTRINGS (1)
#define MICROPY_COMP_MODULE_CONST (1)
#define MICROPY_COMP_TRIPLE_TUPLE_ASSIGN (1)
#define MICROPY_COMP_RETURN_IF_EXPR (1)
#endif

// extmod modules ck_dsl needs (compiled in via build_embed.py).
#define MICROPY_PY_RE (1)
#define MICROPY_PY_RE_MATCH_GROUPS (1)
#define MICROPY_PY_RE_MATCH_SPAN_START_END (1)
#define MICROPY_PY_RE_SUB (1)
#define MICROPY_PY_BUILTINS_BYTEARRAY (1)

// str methods ck_dsl uses that CORE_FEATURES leaves off (pure-str, no deps).
#define MICROPY_PY_BUILTINS_STR_PARTITION (1)
#define MICROPY_PY_BUILTINS_STR_CENTER (1)
#define MICROPY_PY_BUILTINS_STR_COUNT (1)
#define MICROPY_PY_BUILTINS_STR_SPLITLINES (1)

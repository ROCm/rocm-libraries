/* MicroPython embed-port config for the ck-dsl-provider Arch-A integration.
 * Starts from the minimal embed config and adds what ck_dsl codegen needs:
 * arbitrary-precision ints (MPZ), double floats, sys.modules. Frozen modules
 * carry the shims + the transformed ck_dsl codegen closure (no filesystem).
 */
#include <port/mpconfigport_common.h>

// Minimal base (disables optional extmod modules the bare embed port doesn't
// compile); we raise features incrementally as the bundle needs them.
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)

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

// Chain the frozen modules' qstr pool onto the runtime qstr pool, so frozen
// bytecode's qstr indices resolve (else find_qstr asserts). Standard freezing wiring.
#if MICROPY_MODULE_FROZEN_MPY
#define MICROPY_QSTR_EXTRA_POOL mp_qstr_frozen_const_pool
#endif

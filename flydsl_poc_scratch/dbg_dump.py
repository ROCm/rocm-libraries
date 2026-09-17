import flydsl.compiler.jit_function as J

_orig = (
    J.MlirCompiler.compile.__func__
    if hasattr(J.MlirCompiler.compile, "__func__")
    else None
)

# Patch module.operation.verify indirectly: wrap compile to print module before verify.
import types

_real_compile = J.MlirCompiler.compile


def _patched(cls, module, *a, **k):
    try:
        txt = str(module)
    except Exception as e:
        txt = f"<str(module) failed: {e}>"
    with open("module_dump.mlir", "w") as f:
        f.write(txt)
    print("=== wrote module_dump.mlir (%d bytes) ===" % len(txt))
    return _real_compile(module, *a, **k)


J.MlirCompiler.compile = classmethod(_patched)

import rmsnorm

rmsnorm.main()

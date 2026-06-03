# Minimal `os` shim for the embed build (no filesystem). ck_dsl's codegen path
# only needs os.getenv (lower_llvm reads CK_DSL_LLVM_FLAVOR; default is fine).
def getenv(key, default=None):
    return default

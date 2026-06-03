# Stub: ck_dsl's comgr path never spawns processes; only compile_kernel_via_hipcc
# (the unused hipcc backend) would, and it isn't reached in the embed build.
PIPE = -1
STDOUT = -2


class SubprocessError(Exception):
    pass


class CalledProcessError(SubprocessError):
    pass


def run(*args, **kwargs):
    raise OSError("subprocess not available in the embed build")

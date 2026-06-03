# Stub: only the unused hipcc backend uses tempfile; the comgr path never does.
class TemporaryDirectory:
    def __init__(self, *args, **kwargs):
        raise OSError("tempfile not available in the embed build")

    def __enter__(self):
        raise OSError("tempfile not available in the embed build")

    def __exit__(self, *exc):
        return False

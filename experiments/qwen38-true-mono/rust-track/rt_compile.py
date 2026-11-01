"""Compile one ``rt_builds`` label in this (fresh) process and write its code object.

    python rt_compile.py LABEL comgr|direct OUT.hsaco

The direct backend reads ``ROCKE_DIRECT_LEVERS`` from the environment; a refusal is an error (exit 2).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    label, backend, out = sys.argv[1:4]
    import rt_builds
    from rocke.helpers.compile import compile_kernel

    art = compile_kernel(rt_builds.build(label), arch="gfx950", backend="python", code_object_backend=backend)
    if getattr(art, "code_object_backend", backend) != backend:
        print(f"{label}: {backend} refused: {getattr(art, 'direct_fallback', '')}", file=sys.stderr)
        return 2
    with open(out, "wb") as fh:
        fh.write(art.hsaco)
    return 0


if __name__ == "__main__":
    sys.exit(main())

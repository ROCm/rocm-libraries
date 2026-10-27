# Launch-surface fixtures

Two synthetic engine manifests, read by `tests/test_launch_surface.py`. They are
controlled inputs to `launch_surface.py --check`, not descriptions of a shipped
engine; literal expectations in the tests describe these files. Every path they name
(`provider/Mirror.cpp`, `provider/rocke/library/kernels/mirror.py`, `tests/test_mirror.py`)
is written by the test under a temporary check root, so every path and symbol check
runs on a bare checkout and no rocKE or provider file can move them.

To audit a real manifest instead, set (a path that does not exist is an error); the
check then resolves paths against the repository root:

    HIPDNN_INGESTOR_MANIFEST_GFX942=/abs/path/to/gfx942.engine.yaml
    HIPDNN_INGESTOR_MANIFEST_GFX950=/abs/path/to/gfx950.engine.yaml

- `gfx942.engine.yaml`: `grid` and `applicability` share one `cpp_mirror` file and split
  its two required metadata fields, so only the union-over-shared-mirror rule passes
  it; `kernargs` and `spec_resolution` are `guard: none`.
- `gfx950.engine.yaml`: `applicability` alone declares the fields its mirror reads, while
  `kernargs` shares that mirror declaring nothing, so deleting `applicability` is caught
  and deleting `kernargs` is not (`TestUndeclaredSurfaceLimit`); only `spec_resolution`
  is `guard: none`.

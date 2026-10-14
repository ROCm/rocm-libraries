# Launch-surface fixtures

Two synthetic engine manifests, read by `tests/test_launch_surface.py` so it runs on a bare
checkout. They are controlled inputs to `launch_surface.py --check`, not descriptions of a
shipped engine; literal expectations in the tests describe these files. Their mirrors and
`python_source` entries name real files and symbols, so every path and symbol check runs.

To audit a real manifest instead, set (a path that does not exist is an error):

    HIPDNN_INGESTOR_MANIFEST_GFX942=/abs/path/to/gfx942.engine.yaml
    HIPDNN_INGESTOR_MANIFEST_GFX950=/abs/path/to/gfx950.engine.yaml

- `gfx942.engine.yaml`: `grid` and `applicability` share one `cpp_mirror` and split its
  two required metadata fields, so only the union-over-shared-mirror rule passes it;
  `kernargs` and `spec_resolution` are `guard: none`.
- `gfx950.engine.yaml`: `applicability` alone declares the fields its mirror reads, while
  `kernargs` shares that mirror declaring nothing, so deleting `applicability` is caught
  and deleting `kernargs` is not (`TestUndeclaredSurfaceLimit`); only `spec_resolution`
  is `guard: none`.

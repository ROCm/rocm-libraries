# Authoring-profile fixtures

Two synthetic `.profile.yaml` files, shaped like the per-arch profiles an integration
author writes by hand, committed so `tests/test_launch_surface.py` runs on a bare checkout
with nothing set in the environment.

They are **controlled inputs to the tools**, not descriptions of any shipped engine's real
launch surface: a test asserting `set(unguarded) == {"spec_resolution"}` against
`gfx950.profile.yaml` pins what `launch_surface.py --check` does with the input below it,
not what the real gfx950 attention-dense pack leaves unguarded. Read every literal
expectation in those modules that way.

The directory is named for the artifact rather than the tool, because an authoring profile
is one file per architecture that several tools take blocks out of. Only the
launch-surface audits read these today.

## Overriding

Both modules prefer an environment variable over the fixture, so an author can point the
same audits at a real profile:

    HIPDNN_INGESTOR_PROFILE_GFX942=/abs/path/to/gfx942.profile.yaml
    HIPDNN_INGESTOR_PROFILE_GFX950=/abs/path/to/gfx950.profile.yaml

A variable set to a path that does not exist is an error, not a silent fall back to the
fixture, which would read as the author's own profile passing. Pointed at a real profile
the fixture-keyed assertions fail on a different honest set; the classes' docstrings say
which those are.

`tests/test_coverage_gate.py` takes no fixture default: its opt-in class drives the whole
gate against a packed tree an author really built, which no synthetic profile describes.
It accepts the variables above (and unsuffixed `HIPDNN_INGESTOR_PROFILE`) and skips when
none is set.

## `gfx942.profile.yaml`

Four surfaces, two of them (`kernargs`, `spec_resolution`) honestly `guard: none` /
`test: none`. `grid` and `applicability` cite the same `cpp_mirror` and split its two
required metadata fields, so only the audit's union-over-shared-mirror rule makes the
check pass.

## `gfx950.profile.yaml`

Four surfaces, one of them (`spec_resolution`) honestly `guard: none` / `test: none`.
`applicability` is the sole declarer of both fields its mirror reads through a required
accessor, while `kernargs` cites that same mirror declaring nothing, so deleting
`applicability` is caught by the metadata-field scan and deleting `kernargs` is not --
the two branches of the residual gap `TestUndeclaredSurfaceLimit` exercises.

## Why the mirrors are real files

`ConvNative.cpp` is committed and really does read `dtype` and `block_size` through
`getStringMetadata`/`getIntMetadata`, and the `python_source` entries name functions that
exist under `rocke/library`, so every path and symbol check does work here. A fixture
pointing at invented paths would pass the same audit by never reaching those checks.

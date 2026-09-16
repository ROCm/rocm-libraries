# Authoring-profile fixtures

Two synthetic `.profile.yaml` files, shaped like the per-arch profiles an integration
author writes by hand, committed so the profile-driven tests in
`tests/test_launch_surface.py` run on a bare checkout with nothing set in the
environment.

They are **controlled inputs to the tools**, not descriptions of any shipped engine's
real launch surface. A test asserting `set(unguarded) == {"spec_resolution"}` against
`gfx950.profile.yaml` pins what `launch_surface.py --check` does with the input below
it; it says nothing about which surfaces the real gfx950 attention-dense pack leaves
unguarded. Read every literal expectation in those modules that way.

This directory is named for the artifact rather than for the tool that reads it. An
authoring profile is one file per architecture that several tools take blocks out of —
`launch_surface.py` reads `launch_surface:`, other audits read blocks of their own — so
a fixture filed under one tool's name would have to move the first time a second tool
grew a case against it. Only the launch-surface audits read these today, and each
carries the blocks that audit needs and nothing further.

## Overriding

Both modules prefer an environment variable over the fixture, so an author can point
the same audits at their real profile:

    HIPDNN_INGESTOR_PROFILE_GFX942=/abs/path/to/gfx942.profile.yaml
    HIPDNN_INGESTOR_PROFILE_GFX950=/abs/path/to/gfx950.profile.yaml

A variable that is set but does not name an existing file is an error, not a silent
fall back to the fixture — a typo that quietly audited the fixture instead would read
as the author's own profile passing.

The literal expectations described above are the cost of the override: pointed at a
real profile, those assertions describe the fixture and will fail on a different
honest set. That is the intended reading — the failure names the mismatch rather than
hiding it — and the classes' docstrings say which assertions are fixture-keyed.

`tests/test_coverage_gate.py` takes no fixture default. Its opt-in class drives the
whole gate against a packed tree an author really built, and a synthetic profile does
not describe any real pack, so a default there would manufacture a failure on the
first machine that has a build. It accepts the variables above (and unsuffixed
`HIPDNN_INGESTOR_PROFILE`) and skips when none is set.

## `gfx942.profile.yaml`

Four surfaces, two of them (`kernargs`, `spec_resolution`) honestly `guard: none` /
`test: none`. `grid` and `applicability` cite the same `cpp_mirror` and split its two
required metadata fields between them, so the audit's union-over-shared-mirror rule is
what makes the check pass — neither surface's `kmd_fields` covers the file alone.

## `gfx950.profile.yaml`

Four surfaces, one of them (`spec_resolution`) honestly `guard: none` / `test: none`.
`applicability` is the sole declarer of both fields its mirror reads through a required
accessor, and `kernargs` cites that same mirror while declaring nothing. Deleting
`applicability` is therefore caught by the metadata-field scan and deleting `kernargs`
is not — the two branches of the documented residual gap `TestUndeclaredSurfaceLimit`
exercises.

## Why the mirrors are real files

`ConvNative.cpp` is committed and really does read `dtype` and `block_size` through
`getStringMetadata`/`getIntMetadata`, and the `python_source` entries name functions
that really exist under `rocke/library`. Every path and symbol check therefore does
work here. A fixture pointing at invented paths would pass the same audit by never
reaching those checks.

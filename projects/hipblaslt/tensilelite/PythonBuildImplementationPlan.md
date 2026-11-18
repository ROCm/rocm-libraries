<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Implementation Plan: rocm-libraries

Status: Current implementation plan
Decision record: `PythonBuildGrillingDecisions.md` Q095–Q120, revised by Q121–Q134
(2026-11-11)

## Final contract

```text
canonical tensilelite wheel + build-tree rocisa
                    │
                    ▼
           selected build Python
                    │
                    ▼
logic validation, create-library, and ext-op generation
```

`tensilelite-client` is not part of this graph. It remains a separately built
benchmark/validation executable used only by explicit client workflows.

## Implementation

- Default `TENSILELITE_ENABLE_CLIENT` from `TENSILELITE_BUILD_TESTING`; device
  generation permits both TensileLite host and client to be disabled.
- Build the wheel and unpack it into the build tree for every device generation
  path; put it and build-tree rocisa on a command-scoped `PYTHONPATH`. Nothing is
  installed into the build Python (Q129).
- There is no compatibility wheel (Q133). The wheel's version comes from
  `VERSION` plus `release_metadata.py`: `+g<sha>[.dirty]`, or `+rocm`/`+devrocm`
  when TheRock passes its package version (Q121). The same configure records the
  commit for the native client header, generated only when the client target is
  enabled (Q123).
- Validate the selected ROCm installation at package import. Import rocisa only
  when generator code needs it, and resolve and validate the configured or
  standard client only when a caller explicitly asks for its path.
- Preserve `invoke install` as the full editable-development workflow, including
  client build and binding.
- Install the client only through the non-Windows TensileLite test-artifact
  workflow and CMake `tests` component. Q109 records the eventual promotion to
  `rocm[libraries]`; it is not current implementation work.

## Validation

- Unit-test client-free import/help, lazy client lookup diagnostics, and the
  existing editable-development binding flow.
- Configure the restored device presets with TensileLite host/client disabled;
  build a filtered device-library target through the canonical wheel and
  build-tree rocisa.
- Verify wheel generation and build-tree unpacking for device builds, the
  wheel's stamped commit, and client-header generation only for client builds.

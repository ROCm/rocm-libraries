<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Implementation Plan: rocm-libraries

Status: Current implementation plan
Decision record: `PythonBuildGrillingDecisions.md` Q115–Q129

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
- Build and force-install the canonical wheel with `--no-deps` for every device
  generation path. Expose only build-tree rocisa through command-scoped
  `PYTHONPATH`; run package commands directly from the installed wheel.
- Build the compatibility wheel only for release/artifact-test workflows. Keep
  one release identity from `VERSION` plus `release_metadata.py`; generate the
  native client version header only when the client target is enabled.
- Keep rocisa and ROCm-release validation at package import. Resolve and
  validate the configured or standard client only when a caller explicitly asks
  for its path.
- Preserve `invoke install` as the full editable-development workflow, including
  client build and binding.
- Install the client only through the non-Windows TensileLite test-artifact
  workflow and CMake `tests` component. Q129 records the eventual promotion to
  `rocm[libraries]`; it is not current implementation work.

## Validation

- Unit-test client-free import/help, lazy client lookup diagnostics, and the
  existing editable-development binding flow.
- Configure the restored device presets with TensileLite host/client disabled;
  build a filtered device-library target through the canonical wheel and
  build-tree rocisa.
- Verify canonical-wheel generation for device builds, compatibility-wheel
  generation for artifact tests, and client-header generation only for client
  builds.

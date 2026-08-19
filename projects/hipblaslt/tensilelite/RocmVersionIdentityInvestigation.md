<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# ROCm Version Identity Investigation

Status: Final two-adapter runtime model accepted. The intermediary implementation
now uses Python SDK package identity plus core tool trampolines, and conventional
prefixes use root-relative tools plus a base-only `.info/version` identity.
Package-mode client requests deliberately fail until the client ships in
`rocm-sdk-libraries` with its own trampoline.

## Question

TensileLite currently records and validates the base ROCm compatibility identity
in its wheel version:

```text
<TensileLite component version>+rocm<X.Y.Z>
```

The question is whether that identity should instead include the full ROCm
publication version for nightly, RC, or dev installations.

## Intermediary implementation

The current wheel/runtime implementation uses the required `rocm-sdk-core`
package to avoid expanding `rocm[devel]` for version validation:

- `rocm_sdk_core.__version__` supplies the full Python publication identity.
- The active Python environment's `rocm-sdk-core` console-script trampolines
  supply compiler and toolchain commands.
- `rocm_sdk_core.__version__` is the full Python package publication identity;
  `.info/version` remains the non-Python fallback's base compatibility value.

The active Python SDK path therefore compares the full nightly/RC/dev identity
without inspecting a physical core payload root. A package-mode client request
currently reports that it is unavailable; the final model below adds a
`rocm-sdk-libraries` trampoline once that package ships the client.

## Stable release experiment

Environment: fresh temporary Python 3.12 virtual environment.

Install command:

```bash
python -m pip install --no-cache-dir \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[devel]==7.14.0'
```

Observed values:

```text
rocm package version:              7.14.0
rocm-sdk-core package version:     7.14.0
rocm-sdk-devel package version:    7.14.0

rocm_sdk.__version__:              7.14.0
rocm_sdk_core.__version__:         7.14.0

core .info/version:                7.14.0
expanded devel .info/version:      7.14.0
```

For this stable release, package publication identity and native compatibility
identity are identical.

## Nightly experiment

Environment: separate fresh temporary Python 3.12 virtual environment.

Install command:

```bash
python -m pip install --no-cache-dir --pre \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  'rocm[devel]==10.1.0a20260813'
```

Observed values:

```text
rocm package version:              10.1.0a20260813
rocm-sdk-core package version:     10.1.0a20260813
rocm-sdk-devel package version:    10.1.0a20260813

rocm_sdk.__version__:              10.1.0a20260813
rocm_sdk_core.__version__:         10.1.0a20260813

core .info/version:                10.1.0
expanded devel .info/version:      10.1.0
```

The nightly experiment demonstrates two distinct identities:

```text
Python wheel publication identity: 10.1.0a20260813
Native compatibility identity:     10.1.0
```

The core and devel native metadata agree with each other, but both intentionally
omit the nightly date. The Python meta, core, and devel distributions agree with
each other on the full publication version.

## Why this matters

Using `rocm_sdk_core.__version__` directly would compare full Python wheel
publication identity. It works unchanged for stable releases, but differs from
the current TensileLite `+rocm<X.Y.Z>` tag for nightly, RC, and dev releases.

Reading the core payload's `.info/version` compares native base compatibility
identity. It accepts a matching nightly/RC/dev package set on the same `X.Y.Z`
line, but cannot distinguish two different nightly publications on that line.

## Open questions

1. Is base-release compatibility sufficient for released TensileLite wheels, or
   must a wheel reject a different nightly/RC/dev ROCm publication with the same
   `X.Y.Z` base?
2. If full publication identity is required, what is the authoritative full
   version source for a non-Python native ROCm installation? `.info/version`
   deliberately contains only the base value.
3. If TensileLite's `+rocm` tag is changed to encode full publication identity,
   which version formats and compatibility rules are supported for stable, RC,
   nightly, and dev releases?
4. Should the native client continue to use the same full TensileLite
   distribution version as the Python wheel, independently of whichever ROCm
   identity policy is selected?

The original questions were unresolved when first recorded. The accepted runtime
authority split and the remaining native-root limitation are recorded below.

## Follow-up investigation: full publication identity

This follow-up establishes where the full identity is available for TheRock and
for a native `ROCM_PATH` installation. The authority split recorded above is
now accepted; the remaining tag-format decisions stay open.

### TheRock build and runtime

TheRock's build-time authority is `THEROCK_PACKAGE_VERSION`, not
`THEROCK_ROCM_VERSION`:

- [`compute_rocm_package_version.py`](../../../../TheRock/build_tools/compute_rocm_package_version.py)
  reads the base `rocm-version` from `version.json`, then produces the Python
  publication identity. At the current checkout it reproduced:

  ```text
  release type   rocm_package_version
  ------------   -----------------------------------------------
  nightly        10.1.0a20260813
  RC 2           10.1.0rc2
  CI             10.1.0.dev0+0123456789abcdef0123456789abcdef01234567
  ```

- The same source produces distinct native package versions. For example, the
  nightly Debian and RPM forms are `10.1.0~20260813`, while the Python form is
  `10.1.0a20260813`. The full source format rules are in
  [`docs/packaging/versioning.md`](../../../../TheRock/docs/packaging/versioning.md)
  and `compute_rocm_package_version.py`.
- TheRock's artifact CI computes and supplies `THEROCK_PACKAGE_VERSION` to the
  top-level configuration. However, the hipBLASLt subproject declaration in
  [`math-libs/BLAS/CMakeLists.txt`](../../../../TheRock/math-libs/BLAS/CMakeLists.txt)
  currently forwards only `THEROCK_ROCM_VERSION`. A subproject gets only its
  declared `CMAKE_ARGS`, so a full identity must be forwarded explicitly beside
  the existing base-version argument.

TheRock runtime artifacts retain the full Python publication identity in
`share/therock/therock_manifest.json` under `rocm_package_version`, separately
from the base `rocm_version`. This was verified from an official nightly
`rocm-sdk-core==10.1.0a20260813` wheel unpacked at:

```text
/tmp/rocm-nightly-version-investigation/root/_rocm_sdk_core
```

The relevant values were:

```json
{
  "rocm_package_version": "10.1.0a20260813",
  "rocm_version": "10.1.0"
}
```

The wheel's `METADATA` and `rocm_sdk_core.__version__` also reported
`10.1.0a20260813`, while its `.info/version` reported `10.1.0`. Thus, the
manifest is the suitable full-version runtime source for a TheRock root; it
does not require interpreting a toolchain build string.

The downloaded wheel exposes `rocm_sdk_core.__version__` with the full value.
The intermediary package adapter needs no root API: it validates that value and
uses only interpreter-local core tool trampolines. The current public core
package exposes no root or base-version helper.

One compatibility observation needs resolving before exact equality is made a
release contract: stable Python packages are promoted from prerelease packages.
TheRock's promotion tool rewrites its own package and manifest version from an
`a` or `rc` form to the stable base form. An immutable TensileLite/client tag
containing the prerelease value would therefore require corresponding promotion
support, or a final rebuild, before it could match the promoted runtime.

### Native `ROCM_PATH` nightly experiment

To test a conventional native installation independently from the Python-wheel
route, the following official nightly Debian package was downloaded and
unpacked without installing it or changing `PATH`:

```text
repository: https://rocm.nightlies.amd.com/deb/20260612-27392289755/
package:    amdrocm-base7.14_7.14.0~20260612-27392289755_amd64.deb
location:   /tmp/rocm-native-nightly-version-investigation
```

The package control record, which is external to the installed prefix, contains
the complete native publication identity:

```text
Package: amdrocm-base7.14
Version: 7.14.0~20260612-27392289755
```

Its extracted ROCm root is
`/tmp/rocm-native-nightly-version-investigation/root/opt/rocm/core-7.14`.
Inside that root, the only ROCm release marker is:

```text
.info/version = 7.14.0
```

The root contains neither `therock_manifest.json` nor the nightly date/run ID.
Searching its regular files found no `20260612`, `27392289755`, or
`7.14.0~...` value. This directly proves that the full native package identity
cannot be reconstructed from an arbitrary copied or unpacked `ROCM_PATH`.

For a prefix installed by the native package manager, the package database is
an additional source. On this host, resolving `/opt/rocm` yields
`/opt/rocm-7.2.4`; `dpkg-query -S` identifies its `.info/version` owner as
`rocm-core`, and `dpkg-query -W` reports the full installed package version
`7.2.4.70204-93~24.04`. This is a usable Debian-specific fallback, but it does
not work for copied roots, extracted archives, or a different native package
manager. An RPM fallback would likewise need to query the package owning the
resolved marker path rather than assume a fixed package name.

### Accepted final model

1. Do not derive a full identity from `.info/version`, `hipconfig`, or a
   compiler version. They are compatibility/tool build values, not the Python
   publication identity.
2. The Python SDK package adapter treats `rocm_sdk_core.__version__` as the
   exact full identity. It resolves compiler, assembler, bundler, and
   device-enumerator commands through the active interpreter's
   `rocm-sdk-core` console-script trampolines, first in the normal scripts
   directory and then in the platform's Python user-install scripts directory.
   It does not call `get_core_root()`, inspect a package payload directory, or
   consult `ROCM_PATH`. It also does not fall back to an ambient `PATH`
   executable or borrow an individual tool from a conventional prefix when a
   trampoline is absent: executable `PATH` lookup cannot repair the selected
   wheel tool's native dependency closure, and mixing independently selected
   ROCm tools has no compatibility guarantee.
3. When client capability is requested in the Python SDK package adapter, the
   intermediary implementation returns an explicit unavailable-client error:
   `rocm-sdk-libraries` does not yet ship the client. The final state uses that
   package's `tensilelite-client` trampoline, which forwards `--version` and
   all normal client arguments unchanged.
4. The conventional-prefix adapter resolves a physical root from `ROCM_PATH`,
   `/opt/rocm`, or a ROCm tool found on PATH. It uses root-relative tools and
   client location, and treats `.info/version` as authoritative only for the
   base compatibility line. It deliberately cannot distinguish nightly, RC,
   or CI publications sharing that base.
5. Carry the full publication identity explicitly through TheRock's build
   graph with `THEROCK_PACKAGE_VERSION`; use the selected prefix's
   `.info/version` for standalone prefix builds. Keep the client and wheel on
   one complete TensileLite distribution version.

The temporary wheel-tag grammar is:

- `10.1.0a20260813` and `10.1.0rc2` fit the present local tag shape.
- TheRock CI's `10.1.0.dev0+<SHA>` contains a second `+`; it cannot be
  embedded verbatim after the wheel's local-version `+`. TensileLite encodes
  the complete value losslessly as
  `<component>+devrocm10.1.0.dev0.<SHA>` and restores the same canonical
  identity when validating the active Python core SDK.
- Native Debian/RPM identities use `~` forms such as
  `7.14.0~20260612-27392289755`. The current
  `release_metadata.canonical_rocm_version()` rejects that value. Native builds
  therefore pass their root's base `.info/version` into wheel construction and
  use the documented base-only fallback at runtime.

The runtime authority choice is accepted: Python publication identity when a
core SDK is installed, otherwise native base compatibility identity. The native
fallback remains unable to reject a different nightly, RC, or CI publication on
the same base line.

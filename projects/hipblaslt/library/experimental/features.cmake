# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The experimental API registry. Every hipblaslt-experimental-<feature>.h must have an entry
# here; configure fails once a feature is past its last release. Policy:
# docs/reference/experimental-api.rst. Helpers: cmake/hipblaslt_experimental.cmake.
#
# A feature's revision is not recorded here: it is the HIPBLASLT_EXPERIMENTAL_<NAME> macro in
# its header, where tools/scripts/check_experimental_api.py reads it.

# A feature may ship in at most this many ROCm releases (cmake/rocm_feature_releases.cmake).
set(HIPBLASLT_EXPERIMENTAL_MAX_RELEASES 4)

# At most this many features may be experimental at once.
set(HIPBLASLT_EXPERIMENTAL_MAX_FEATURES 3)

# SPIKE ONLY: a trivial feature that exercises the deadline and revision-bump checks.
hipblaslt_experimental_feature(DEMO
    HEADER       hipblaslt-experimental-demo.h
    OWNER        "@talumbau"
    TRACKING     "https://github.com/ROCm/rocm-libraries/pull/14065"
    INTRODUCED   10.3
    LAST_RELEASE 11.0
)

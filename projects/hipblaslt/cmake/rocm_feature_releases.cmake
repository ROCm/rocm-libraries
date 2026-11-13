# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Ordered list of ROCm feature releases (major.minor; patch releases don't count). Experimental
# API deadlines are counted on this list: "four releases" means four consecutive entries.
# Append each release when it is planned; never reorder or remove entries.
#
# SPIKE: the RFC proposes keeping this list in one shared place for all of ROCm. The entries
# after 10.3 are illustrative.
set(ROCM_FEATURE_RELEASES
    10.3
    10.4
    10.5
    11.0
    11.1
    11.2
)

# The ROCm release this build targets. TheRock passes its ROCM_VERSION; a standalone build uses
# this default, which is updated when a release branch is cut.
set(HIPBLASLT_ROCM_RELEASE "10.3" CACHE STRING
    "ROCm release (major.minor) this hipBLASLt build targets; used for experimental API deadlines")

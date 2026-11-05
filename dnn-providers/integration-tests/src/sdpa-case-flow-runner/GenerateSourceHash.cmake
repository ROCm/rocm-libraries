# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Write a header that defines SDPA_CASE_FLOW_RUNNER_SOURCE_HASH: the first 12 hex digits of
# the sha256 over the concatenated sha256 digests of SOURCE_1 to SOURCE_5.
# Run with cmake -DOUTPUT=... -DSOURCE_1=... (to -DSOURCE_5=...) -P <this file>.

set(_digests "")
foreach(_source IN ITEMS "${SOURCE_1}" "${SOURCE_2}" "${SOURCE_3}" "${SOURCE_4}" "${SOURCE_5}")
    file(SHA256 "${_source}" _digest)
    string(APPEND _digests "${_digest}")
endforeach()
string(SHA256 _combined "${_digests}")
string(SUBSTRING "${_combined}" 0 12 _short)
file(WRITE "${OUTPUT}"
    "#pragma once\n#define SDPA_CASE_FLOW_RUNNER_SOURCE_HASH \"${_short}\"\n")

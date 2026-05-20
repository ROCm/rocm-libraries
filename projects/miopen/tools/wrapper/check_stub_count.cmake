# Q1/Q4 drift check for RFC 0001 (Phase 1 investigation).
#
# Three artifacts must agree on the number of public C entry points:
#
#   HEADER  = include/miopen/miopen.h               (source of truth)
#   RENAME  = src/private/miopen_private_rename.h   (one #define per entry)
#   SRC     = src/private/wrapper.cpp               (one extern "C" stub per entry)
#
# All three are hand-maintained (RFC §6 Q1, Q4 — generators retired). The
# only failure mode hand-authoring introduces is forgetting to update one of
# the three when the public surface changes; this check catches it.
#
# Counting strategy:
#  - HEADER: lines whose first non-whitespace token is `MIOPEN_EXPORT`.
#    Verified empirically (commit a827879e67) to match the parser-based
#    count of distinct entry points exactly.
#  - RENAME: lines starting with `#define miopen` — the per-entry rename
#    lines, distinguishable from the include-guard `#define`.
#  - SRC: occurrences of a `{` alone on a column-0 line. Each generated /
#    hand-written stub formats its body opener that way; forward
#    declarations end with `;` and never produce that pattern.
#
# Invoked by CTest via:
#   cmake -DHEADER=... -DRENAME=... -DSRC=... -P check_stub_count.cmake

foreach(v HEADER RENAME SRC)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "check_stub_count.cmake: ${v} is required")
    endif()
endforeach()

# Source-of-truth count from miopen.h.
file(STRINGS "${HEADER}" _header_lines REGEX "^[ \t]*MIOPEN_EXPORT")
list(LENGTH _header_lines _expected_count)

# Rename count.
file(STRINGS "${RENAME}" _rename_lines REGEX "^#define miopen")
list(LENGTH _rename_lines _rename_count)

# Stub count.
file(READ "${SRC}" _src)
string(REGEX MATCHALL "\n{\n" _stub_braces "${_src}")
list(LENGTH _stub_braces _stub_count)

set(_fail FALSE)
if(NOT _rename_count EQUAL _expected_count)
    message(SEND_ERROR
        "Drift FAIL: ${RENAME} has ${_rename_count} `#define miopen` lines, "
        "expected ${_expected_count} (from MIOPEN_EXPORT count in ${HEADER})")
    set(_fail TRUE)
endif()
if(NOT _stub_count EQUAL _expected_count)
    message(SEND_ERROR
        "Drift FAIL: ${SRC} has ${_stub_count} extern \"C\" stubs, "
        "expected ${_expected_count} (from MIOPEN_EXPORT count in ${HEADER})")
    set(_fail TRUE)
endif()

if(_fail)
    message(FATAL_ERROR "check_stub_count.cmake: drift detected; see SEND_ERROR messages above")
endif()

message(STATUS
    "Drift PASS: ${_expected_count} entry points in ${HEADER}, "
    "${_rename_count} renames in ${RENAME}, ${_stub_count} stubs in ${SRC}")

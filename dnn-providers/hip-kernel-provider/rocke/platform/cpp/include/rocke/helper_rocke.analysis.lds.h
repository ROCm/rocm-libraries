/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.analysis.lds.h -- C++ twin of the CPU-only LDS conflict
 * predictor rocke.analysis.lds (python/rocke/analysis/lds). Host-side only:
 * nothing here emits IR. The C++ engine's LDS planners call this one copy of
 * the predictor instead of keeping their own.
 *
 *   Python (rocke.analysis.lds)          C++ (this header)
 *   ----------------------------------   ---------------------------------------
 *   opcodes._OPCODE_SPECS (OpcodeSpec)   rocke_lds_opcode_spec_t rows:
 *                                        rocke_lds_num_opcodes / _opcode_at
 *   opcodes.get_opcode_spec              rocke_lds_get_opcode_spec
 *   profiles.BUILTIN_PROFILES            rocke_lds_profile_t rows:
 *                                        rocke_lds_num_profiles / _profile_at
 *   <profile>.phase_key                  rocke_lds_phase_key
 *   <profile>.collision_key              rocke_lds_collision_key
 *   registry.registered_targets          rocke_lds_registered_targets
 *   registry.resolve_profile             rocke_lds_resolve_profile
 *   LdsAccess                            rocke_lds_access_t
 *   predict.predict_lds_conflicts        rocke_lds_predict (the summary only)
 *   ConflictSummary                      rocke_lds_summary_t
 *
 * The profile table (cpp/analysis/lds/profiles.cpp) restates the Python
 * profiles, since they are Python classes and not a data file. The CTest
 * rocke_lds_profile_engine_parity (tests/analysis/
 * test_lds_profile_engine_parity.py) pins the two: it reads the C++ table as
 * text and compares every row with the Python registry, and it runs
 * rocke_lds_profile_dump (every profile's phase keys, collision keys and a
 * set of predictions and refusals) against the same dump from Python. A new
 * or changed Python profile fails it until this twin follows.
 *
 * Not ported: the per-access results, the conflict groups and the JSON
 * serialization (model.py, serialization.py); the C++ callers need only the
 * summary.
 */
#ifndef ROCKE_HELPER_ROCKE_ANALYSIS_LDS_H
#define ROCKE_HELPER_ROCKE_ANALYSIS_LDS_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* OpcodeSpec. alignment_bytes is 0 for Python's None (use the width). */
typedef struct rocke_lds_opcode_spec
{
    const char* opcode;
    const char* direction; /* "read" or "write" */
    int access_width_bytes;
    int alignment_bytes;
} rocke_lds_opcode_spec_t;

/* The phase_key / collision_key rules of a profile class. Gfx942Profile
 * inherits Gfx90aProfile's rules; Gfx950Profile overrides the 8- and 16-byte
 * reads; Gfx1201Profile has its own rules and validate_request. */
typedef enum rocke_lds_rules
{
    ROCKE_LDS_RULES_GFX90A = 0,
    ROCKE_LDS_RULES_GFX950 = 1,
    ROCKE_LDS_RULES_GFX1201 = 2
} rocke_lds_rules_t;

/* One profile: identity, supported_wave_sizes, lds_capacity_bytes,
 * supported_opcodes and its rules. */
typedef struct rocke_lds_profile
{
    const char* target;
    int profile_version;
    const int* supported_wave_sizes;
    size_t num_wave_sizes;
    int lds_capacity_bytes;
    const char* const* supported_opcodes;
    size_t num_opcodes;
    rocke_lds_rules_t rules;
} rocke_lds_profile_t;

/* LdsAccess, without the coordinate (it does not affect a prediction). */
typedef struct rocke_lds_access
{
    int access_id;
    int lane;
    int lds_byte_address;
    int access_width_bytes;
    bool active;
} rocke_lds_access_t;

/* ConflictSummary. */
typedef struct rocke_lds_summary
{
    int active_access_count;
    int conflicted_access_count;
    int broadcast_access_count;
    int inactive_access_count;
    int conflict_group_count;
    int maximum_multiplicity;
} rocke_lds_summary_t;

/* The longest phase key any profile returns. */
#define ROCKE_LDS_PHASE_KEY_MAX 4

/* _OPCODE_SPECS, in its order. */
size_t rocke_lds_num_opcodes(void);
const rocke_lds_opcode_spec_t* rocke_lds_opcode_at(size_t i);
/* get_opcode_spec after normalize_opcode (strip, lower); NULL when the opcode
 * is not supported. */
const rocke_lds_opcode_spec_t* rocke_lds_get_opcode_spec(const char* opcode);

/* BUILTIN_PROFILES, in its order. */
size_t rocke_lds_num_profiles(void);
const rocke_lds_profile_t* rocke_lds_profile_at(size_t i);
/* registered_targets(): the targets, sorted. Writes up to `cap` names into
 * `out` (may be NULL) and returns how many there are. */
size_t rocke_lds_registered_targets(const char** out, size_t cap);
/* resolve_profile(target); NULL when no profile has that exact target. */
const rocke_lds_profile_t* rocke_lds_resolve_profile(const char* target);
bool rocke_lds_profile_supports_wave_size(const rocke_lds_profile_t* p, int wave_size);
/* `opcode` is a canonical spelling. */
bool rocke_lds_profile_supports_opcode(const rocke_lds_profile_t* p, const char* opcode);

/* phase_key(opcode, lane): writes the key into `key` and returns its length;
 * -1 when the profile does not support the opcode. */
int rocke_lds_phase_key(const rocke_lds_profile_t* p,
                        const char* opcode,
                        int lane,
                        int key[ROCKE_LDS_PHASE_KEY_MAX]);
/* collision_key(opcode, lds_byte_address); -1 when the profile does not
 * support the opcode. */
int rocke_lds_collision_key(const rocke_lds_profile_t* p, const char* opcode, int lds_byte_address);

/* predict_lds_conflicts(target=, opcode=, wave_size=, accesses=).summary.
 * On a refused request returns ROCKE_ERR_VALUE and writes the Python
 * ValueError message into `msg` (truncated to `msg_cap` - 1; may be NULL). */
rocke_status_t rocke_lds_predict(const char* target,
                                 const char* opcode,
                                 int wave_size,
                                 const rocke_lds_access_t* accesses,
                                 size_t num_accesses,
                                 rocke_lds_summary_t* summary,
                                 char* msg,
                                 size_t msg_cap);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++ API for the engine's planners: throws ckc::ValueError with the Python
 * message where Python raises (LdsPredictionError, UnsupportedLdsTargetError,
 * UnsupportedLdsOpcodeError and ModelValidationError are ValueErrors). */
#include <string>
#include <vector>

namespace rocke_analysis_lds
{
/* registered_targets() */
std::vector<std::string> registered_targets();
/* resolve_profile(target) */
const rocke_lds_profile_t& resolve_profile(const std::string& target);
/* LdsAccess(...) for each access, then predict_lds_conflicts(...).summary */
rocke_lds_summary_t predict_lds_conflicts(const std::string& target,
                                          const std::string& opcode,
                                          int wave_size,
                                          const std::vector<rocke_lds_access_t>& accesses);
} // namespace rocke_analysis_lds
#endif

#endif /* ROCKE_HELPER_ROCKE_ANALYSIS_LDS_H */

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ twin of rocke.analysis.lds.registry and rocke.analysis.lds.predict:
 * exact target selection and the conflict prediction of one LDS instruction
 * in one wave. See rocke/helper_rocke.analysis.lds.h for the mapping.
 *
 * Every Python raise is one ckc::ValueError with the same message; the C
 * entry point catches it.
 */
#include "rocke/helper_rocke.analysis.lds.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <new>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rocke/error.hpp"

namespace rocke_analysis_lds
{

/* {s!r} */
static std::string repr(const std::string& s)
{
    size_t n = rocke_py_str_repr(s.c_str(), NULL, 0);
    std::string out(n + 1, '\0');
    rocke_py_str_repr(s.c_str(), &out[0], out.size());
    out.resize(n);
    return out;
}

static std::string join(const std::vector<std::string>& items)
{
    std::string out;
    for(size_t i = 0; i < items.size(); ++i)
        out += (i ? ", " : "") + items[i];
    return out;
}

/* registry.registered_targets(): the targets, sorted. */
std::vector<std::string> registered_targets()
{
    std::vector<std::string> out;
    for(size_t i = 0; i < rocke_lds_num_profiles(); ++i)
        out.push_back(rocke_lds_profile_at(i)->target);
    std::sort(out.begin(), out.end());
    return out;
}

/* registry.resolve_profile(target) */
const rocke_lds_profile_t& resolve_profile(const std::string& target)
{
    const rocke_lds_profile_t* p = rocke_lds_resolve_profile(target.c_str());
    if(p == NULL)
        throw ckc::ValueError("unsupported LDS target " + repr(target)
                              + "; registered targets: " + join(registered_targets()));
    return *p;
}

/* opcodes.get_opcode_spec(opcode) */
static const rocke_lds_opcode_spec_t& get_opcode_spec(const std::string& opcode)
{
    const rocke_lds_opcode_spec_t* spec = rocke_lds_get_opcode_spec(opcode.c_str());
    if(spec == NULL)
    {
        std::vector<std::string> choices;
        for(size_t i = 0; i < rocke_lds_num_opcodes(); ++i)
            choices.push_back(rocke_lds_opcode_at(i)->opcode);
        throw ckc::ValueError("unsupported LDS opcode " + repr(opcode)
                              + "; supported opcodes: " + join(choices));
    }
    return *spec;
}

/* model._require_int(name, value, minimum=minimum) on an int */
static void require_at_least(const char* name, int value, int minimum)
{
    if(value < minimum)
        throw ckc::ValueError(std::string(name) + " must be at least " + std::to_string(minimum));
}

/* Gfx1201Profile.validate_request; the other profiles have none. */
static void validate_request(const rocke_lds_profile_t& p,
                             const rocke_lds_opcode_spec_t& spec,
                             const std::vector<rocke_lds_access_t>& ordered)
{
    if(p.rules != ROCKE_LDS_RULES_GFX1201)
        return;
    std::vector<const rocke_lds_access_t*> active;
    for(const rocke_lds_access_t& a : ordered)
        if(a.active)
            active.push_back(&a);
    if(active.size() > 2)
        throw ckc::ValueError("gfx1201 v1 supports at most two active lanes");
    if(active.size() == 2)
    {
        int distance = std::abs(active[0]->lds_byte_address - active[1]->lds_byte_address);
        if(distance < spec.access_width_bytes)
            throw ckc::ValueError("gfx1201 v1 requires distinct, non-overlapping addresses");
    }
}

/* predict._validate_accesses: the accesses sorted by access_id. */
static std::vector<rocke_lds_access_t> validate_accesses(const rocke_lds_profile_t& p,
                                                         const rocke_lds_opcode_spec_t& spec,
                                                         int wave_size,
                                                         std::vector<rocke_lds_access_t> acc)
{
    const std::string target = p.target;
    if(!rocke_lds_profile_supports_wave_size(&p, wave_size))
    {
        std::vector<int> sizes(p.supported_wave_sizes, p.supported_wave_sizes + p.num_wave_sizes);
        std::sort(sizes.begin(), sizes.end());
        std::vector<std::string> choices;
        for(int s : sizes)
            choices.push_back(std::to_string(s));
        throw ckc::ValueError("unsupported wave_size " + std::to_string(wave_size) + " for "
                              + target + "; supported wave sizes: " + join(choices));
    }
    if(!rocke_lds_profile_supports_opcode(&p, spec.opcode))
        throw ckc::ValueError("opcode " + repr(spec.opcode) + " is not supported by " + target);

    std::stable_sort(acc.begin(), acc.end(), [](const auto& a, const auto& b) {
        return a.access_id < b.access_id;
    });
    for(size_t i = 1; i < acc.size(); ++i)
        if(acc[i].access_id == acc[i - 1].access_id)
            throw ckc::ValueError("access_id values must be unique");
    std::set<int> lanes;
    for(const rocke_lds_access_t& a : acc)
        if(a.active && !lanes.insert(a.lane).second)
            throw ckc::ValueError("each request allows only one active access per lane");

    const int width = spec.access_width_bytes;
    for(const rocke_lds_access_t& a : acc)
    {
        const std::string id = std::to_string(a.access_id);
        if(a.lane >= wave_size)
            throw ckc::ValueError("access " + id + " lane " + std::to_string(a.lane)
                                  + " must be smaller than wave_size " + std::to_string(wave_size));
        if(a.access_width_bytes != width)
            throw ckc::ValueError("access " + id + " width " + std::to_string(a.access_width_bytes)
                                  + " does not match " + spec.opcode + " width "
                                  + std::to_string(width));
        /* Python ints do not overflow: compare in 64 bits. */
        long long end = (long long)a.lds_byte_address + a.access_width_bytes;
        if(end > p.lds_capacity_bytes)
            throw ckc::ValueError("access " + id + " end address " + std::to_string(end)
                                  + " exceeds " + target + " LDS capacity "
                                  + std::to_string(p.lds_capacity_bytes));
        int alignment = spec.alignment_bytes ? spec.alignment_bytes : width;
        if(a.lds_byte_address % alignment)
            throw ckc::ValueError("access " + id + " address " + std::to_string(a.lds_byte_address)
                                  + " must be " + std::to_string(alignment) + "-byte aligned");
    }
    /* Profiles can narrow the common request contract to their observed scope. */
    validate_request(p, spec, acc);
    return acc;
}

/* predict_lds_conflicts(...).summary: _build_groups, then
 * ConflictSummary.from_results. */
rocke_lds_summary_t predict_lds_conflicts(const std::string& target,
                                          const std::string& opcode,
                                          int wave_size,
                                          const std::vector<rocke_lds_access_t>& accesses)
{
    /* LdsAccess.__post_init__, one access at a time */
    for(const rocke_lds_access_t& a : accesses)
    {
        require_at_least("access_id", a.access_id, 0);
        require_at_least("lane", a.lane, 0);
        require_at_least("lds_byte_address", a.lds_byte_address, 0);
        require_at_least("access_width_bytes", a.access_width_bytes, 1);
    }
    const rocke_lds_profile_t& p = resolve_profile(target);
    const rocke_lds_opcode_spec_t& spec = get_opcode_spec(opcode);
    std::vector<rocke_lds_access_t> ordered = validate_accesses(p, spec, wave_size, accesses);

    /* key: (phase key, collision key) -> the members' addresses */
    std::map<std::pair<std::vector<int>, int>, std::vector<int>> buckets;
    for(const rocke_lds_access_t& a : ordered)
    {
        if(!a.active)
            continue;
        int key[ROCKE_LDS_PHASE_KEY_MAX];
        int n = rocke_lds_phase_key(&p, spec.opcode, a.lane, key);
        std::vector<int> phase(key, key + n);
        buckets[{phase, rocke_lds_collision_key(&p, spec.opcode, a.lds_byte_address)}].push_back(
            a.lds_byte_address);
    }
    const bool write = strcmp(spec.direction, "write") == 0;
    rocke_lds_summary_t s = {};
    for(const auto& kv : buckets)
    {
        const std::vector<int>& m = kv.second;
        if(m.size() < 2)
            continue;
        bool one_addr = std::all_of(m.begin(), m.end(), [&](int x) { return x == m[0]; });
        if(one_addr)
        {
            if(write)
                continue;
            s.broadcast_access_count += (int)m.size();
            continue;
        }
        s.conflicted_access_count += (int)m.size();
        ++s.conflict_group_count;
        s.maximum_multiplicity = std::max(s.maximum_multiplicity, (int)m.size());
    }
    for(const rocke_lds_access_t& a : ordered)
        a.active ? ++s.active_access_count : ++s.inactive_access_count;
    return s;
}

} // namespace rocke_analysis_lds

extern "C" {

size_t rocke_lds_registered_targets(const char** out, size_t cap)
{
    std::vector<const char*> names;
    for(size_t i = 0; i < rocke_lds_num_profiles(); ++i)
        names.push_back(rocke_lds_profile_at(i)->target);
    std::sort(
        names.begin(), names.end(), [](const char* a, const char* b) { return strcmp(a, b) < 0; });
    for(size_t i = 0; out != NULL && i < names.size() && i < cap; ++i)
        out[i] = names[i];
    return names.size();
}

const rocke_lds_profile_t* rocke_lds_resolve_profile(const char* target)
{
    for(size_t i = 0; target != NULL && i < rocke_lds_num_profiles(); ++i)
        if(strcmp(rocke_lds_profile_at(i)->target, target) == 0)
            return rocke_lds_profile_at(i);
    return NULL;
}

rocke_status_t rocke_lds_predict(const char* target,
                                 const char* opcode,
                                 int wave_size,
                                 const rocke_lds_access_t* accesses,
                                 size_t num_accesses,
                                 rocke_lds_summary_t* summary,
                                 char* msg,
                                 size_t msg_cap)
{
    if(msg != NULL && msg_cap > 0)
        msg[0] = '\0';
    try
    {
        std::vector<rocke_lds_access_t> acc(accesses, accesses + (accesses ? num_accesses : 0));
        rocke_lds_summary_t s = rocke_analysis_lds::predict_lds_conflicts(
            target ? target : "", opcode ? opcode : "", wave_size, acc);
        if(summary != NULL)
            *summary = s;
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        if(msg != NULL && msg_cap > 0)
            snprintf(msg, msg_cap, "%s", e.what());
        return e.code();
    }
    catch(const std::bad_alloc&)
    {
        if(msg != NULL && msg_cap > 0)
            snprintf(msg, msg_cap, "%s", "out of memory");
        return ROCKE_ERR_OOM;
    }
}

} // extern "C"

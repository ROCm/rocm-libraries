// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The C++ side of rocke_lds_profile_engine_parity: answers requests about the
// C++ twin of rocke.analysis.lds (rocke/helper_rocke.analysis.lds.h), one per
// stdin line, so test_lds_profile_engine_parity.py can compare each answer
// with the Python registry's. The Python test builds the requests; this
// program only reads them.
//
//   table                      the opcode table, the sorted targets and every
//                              profile's facts
//   keys TARGET OPCODE         phase_key over lanes 0..63 and collision_key
//                              over addresses 0..1023
//   predict TARGET OPCODE WAVE [ID:LANE:ADDR:WIDTH:ACTIVE ...]
//                              predict_lds_conflicts(...).summary, or the
//                              ValueError message
//
// Output lines use the same spelling test_lds_profile_engine_parity.py writes
// for Python.
#include "rocke/helper_rocke.analysis.lds.h"

#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string join_ints(std::vector<int> v)
{
    std::string out;
    for(size_t i = 0; i < v.size(); ++i)
        out += (i ? "," : "") + std::to_string(v[i]);
    return out;
}

static void table()
{
    for(size_t i = 0; i < rocke_lds_num_opcodes(); ++i)
    {
        const rocke_lds_opcode_spec_t* s = rocke_lds_opcode_at(i);
        printf("opcode %s %s %d %s\n",
               s->opcode,
               s->direction,
               s->access_width_bytes,
               s->alignment_bytes ? std::to_string(s->alignment_bytes).c_str() : "None");
    }
    size_t n = rocke_lds_registered_targets(NULL, 0);
    std::vector<const char*> targets(n);
    rocke_lds_registered_targets(targets.data(), n);
    std::string line = "targets";
    for(const char* t : targets)
        line += std::string(" ") + t;
    printf("%s\n", line.c_str());
    for(size_t i = 0; i < rocke_lds_num_profiles(); ++i)
    {
        const rocke_lds_profile_t* p = rocke_lds_profile_at(i);
        std::vector<int> waves(p->supported_wave_sizes,
                               p->supported_wave_sizes + p->num_wave_sizes);
        std::sort(waves.begin(), waves.end());
        std::vector<std::string> ops(p->supported_opcodes, p->supported_opcodes + p->num_opcodes);
        std::sort(ops.begin(), ops.end());
        std::string opl;
        for(size_t j = 0; j < ops.size(); ++j)
            opl += (j ? "," : "") + ops[j];
        printf("profile %s version=%d wave_sizes=%s capacity=%d opcodes=%s\n",
               p->target,
               p->profile_version,
               join_ints(waves).c_str(),
               p->lds_capacity_bytes,
               opl.c_str());
    }
}

static void keys(const std::string& target, const std::string& opcode)
{
    const rocke_lds_profile_t* p = rocke_lds_resolve_profile(target.c_str());
    if(p == NULL)
    {
        printf("keys %s %s: no profile\n", target.c_str(), opcode.c_str());
        return;
    }
    std::string phases;
    for(int lane = 0; lane < 64; ++lane)
    {
        int key[ROCKE_LDS_PHASE_KEY_MAX];
        int n = rocke_lds_phase_key(p, opcode.c_str(), lane, key);
        if(n < 0)
        {
            printf("keys %s %s: unsupported\n", target.c_str(), opcode.c_str());
            return;
        }
        phases += (lane ? " " : "") + join_ints(std::vector<int>(key, key + n));
    }
    std::vector<int> classes;
    for(int addr = 0; addr < 1024; ++addr)
        classes.push_back(rocke_lds_collision_key(p, opcode.c_str(), addr));
    printf("phase %s %s %s\n", target.c_str(), opcode.c_str(), phases.c_str());
    printf("class %s %s %s\n", target.c_str(), opcode.c_str(), join_ints(classes).c_str());
}

static void predict(const std::string& target,
                    const std::string& opcode,
                    int wave,
                    const std::vector<rocke_lds_access_t>& acc,
                    const std::string& label)
{
    rocke_lds_summary_t s;
    char msg[2048];
    rocke_status_t st = rocke_lds_predict(
        target.c_str(), opcode.c_str(), wave, acc.data(), acc.size(), &s, msg, sizeof(msg));
    if(st != ROCKE_OK)
        printf("predict %s: %s: %s\n",
               label.c_str(),
               st == ROCKE_ERR_VALUE ? "ValueError" : "Error",
               msg);
    else
        printf("predict %s: %d %d %d %d %d %d\n",
               label.c_str(),
               s.active_access_count,
               s.conflicted_access_count,
               s.broadcast_access_count,
               s.inactive_access_count,
               s.conflict_group_count,
               s.maximum_multiplicity);
}

int main()
{
    std::string line;
    while(std::getline(std::cin, line))
    {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if(cmd == "table")
            table();
        else if(cmd == "keys")
        {
            std::string target, opcode;
            in >> target >> opcode;
            keys(target, opcode);
        }
        else if(cmd == "predict")
        {
            std::string target, opcode, tok;
            int wave = 0;
            in >> target >> opcode >> wave;
            std::vector<rocke_lds_access_t> acc;
            while(in >> tok)
            {
                rocke_lds_access_t a = {};
                int active = 1;
                if(sscanf(tok.c_str(),
                          "%d:%d:%d:%d:%d",
                          &a.access_id,
                          &a.lane,
                          &a.lds_byte_address,
                          &a.access_width_bytes,
                          &active)
                   != 5)
                {
                    fprintf(stderr, "bad access %s\n", tok.c_str());
                    return 2;
                }
                a.active = active != 0;
                acc.push_back(a);
            }
            predict(target, opcode, wave, acc, line.substr(8));
        }
        else if(!cmd.empty())
        {
            fprintf(stderr, "unknown command %s\n", cmd.c_str());
            return 2;
        }
    }
    return 0;
}

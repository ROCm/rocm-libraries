/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Selection sweep driver: runs the problems of a spec file (gen_corpus.py) through
// hiptensorCreatePlan with HIPTENSOR_ALGO_DEFAULT_PATIENT on the rocKE backend, so that the
// library times every candidate (HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all) and appends the timings and
// features to HIPTENSOR_ROCKE_SELECT_DUMP. Also, with --default, creates DEFAULT plans and prints
// the selection log (evaluation of the shipped selector without timing).
//
// usage: sweep_driver SPECS --start I --end J --deadline SECONDS --progress FILE [--default]
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    hiptensorDataType_t dtypeOf(std::string const& s)
    {
        if(s == "f16")
            return HIPTENSOR_R_16F;
        if(s == "bf16")
            return HIPTENSOR_R_16BF;
        if(s == "f32")
            return HIPTENSOR_R_32F;
        if(s == "f64")
            return HIPTENSOR_R_64F;
        if(s == "c32")
            return HIPTENSOR_C_32F;
        return HIPTENSOR_C_64F;
    }
    hiptensorComputeDescriptor_t computeOf(std::string const& s)
    {
        if(s == "16F")
            return HIPTENSOR_COMPUTE_DESC_16F;
        if(s == "16BF")
            return HIPTENSOR_COMPUTE_DESC_16BF;
        if(s == "64F")
            return HIPTENSOR_COMPUTE_DESC_64F;
        if(s == "C32F")
            return HIPTENSOR_COMPUTE_DESC_C32F;
        if(s == "C64F")
            return HIPTENSOR_COMPUTE_DESC_C64F;
        return HIPTENSOR_COMPUTE_DESC_32F;
    }
    hiptensorOperator_t opOf(std::string const& s)
    {
        if(s == "MAX")
            return HIPTENSOR_OP_MAX;
        if(s == "MIN")
            return HIPTENSOR_OP_MIN;
        if(s == "MUL")
            return HIPTENSOR_OP_MUL;
        return HIPTENSOR_OP_ADD;
    }

    struct T
    {
        std::vector<int32_t> modes;
        std::vector<int64_t> ext, str;
    };

    T readTensor(std::istringstream& in, int n)
    {
        T t;
        t.modes.resize(n);
        t.ext.resize(n);
        t.str.resize(n);
        for(auto& m : t.modes)
            in >> m;
        for(auto& e : t.ext)
            in >> e;
        for(auto& s : t.str)
            in >> s;
        return t;
    }

    hiptensorTensorDescriptor_t describe(hiptensorHandle_t h, T const& t, hiptensorDataType_t type)
    {
        hiptensorTensorDescriptor_t d = nullptr;
        if(hiptensorCreateTensorDescriptor(h, &d, t.ext.size(), t.ext.data(), t.str.data(), type, 0)
           != HIPTENSOR_STATUS_SUCCESS)
            return nullptr;
        return d;
    }

    std::string lastSelection;
    void        callback(int32_t, char const*, char const* msg)
    {
        if(std::strstr(msg, "rocKE selection:") != nullptr)
            lastSelection = msg;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(
            stderr,
            "usage: %s SPECS [--start I] [--end J] [--deadline S] [--progress F] [--default]\n",
            argv[0]);
        return 2;
    }
    std::string specs = argv[1], progress;
    long        start = 0, end = 1L << 40;
    double      deadline   = 1e9;
    bool        defaultRun = false;
    for(int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        if(a == "--start")
            start = std::atol(argv[++i]);
        else if(a == "--end")
            end = std::atol(argv[++i]);
        else if(a == "--deadline")
            deadline = std::atof(argv[++i]);
        else if(a == "--progress")
            progress = argv[++i];
        else if(a == "--default")
            defaultRun = true;
    }
    auto              t0 = std::chrono::steady_clock::now();
    hiptensorHandle_t h  = nullptr;
    if(hiptensorCreate(&h) != HIPTENSOR_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "hiptensorCreate failed\n");
        return 3;
    }
    hiptensorLoggerSetCallback(&callback);
    hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE | HIPTENSOR_LOG_LEVEL_ERROR);

    std::ifstream in(specs);
    std::string   line;
    long          index = -1, done = 0, failed = 0;
    while(std::getline(in, line))
    {
        if(line.empty())
            continue;
        index++;
        if(index < start || index >= end)
            continue;
        double elapsed
            = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if(elapsed > deadline)
        {
            std::printf("DEADLINE next=%ld\n", index);
            break;
        }
        std::istringstream                       ss(line);
        std::string                              kind, dtype, compute;
        long                                     id   = 0;
        hiptensorOperationDescriptor_t           desc = nullptr;
        std::vector<hiptensorTensorDescriptor_t> descs;
        ss >> kind >> id >> dtype >> compute;
        auto              type = dtypeOf(dtype);
        hiptensorStatus_t st   = HIPTENSOR_STATUS_SUCCESS;
        if(kind == "C")
        {
            int bil = 0, na = 0, nb = 0, nd = 0;
            ss >> bil >> na >> nb >> nd;
            T    A = readTensor(ss, na), B = readTensor(ss, nb), D = readTensor(ss, nd);
            auto dA = describe(h, A, type), dB = describe(h, B, type), dD = describe(h, D, type);
            auto dC = bil ? describe(h, D, type) : nullptr;
            descs   = {dA, dB, dD, dC};
            st      = hiptensorCreateContraction(h,
                                            &desc,
                                            dA,
                                            A.modes.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            dB,
                                            B.modes.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            dC,
                                            bil ? D.modes.data() : nullptr,
                                            HIPTENSOR_OP_IDENTITY,
                                            dD,
                                            D.modes.data(),
                                            computeOf(compute));
        }
        else
        {
            std::string op;
            int         beta = 0, na = 0, nd = 0;
            ss >> op >> beta >> na >> nd;
            T    A = readTensor(ss, na), D = readTensor(ss, nd);
            auto dA = describe(h, A, type), dC = describe(h, D, type), dD = describe(h, D, type);
            descs = {dA, dC, dD};
            st    = hiptensorCreateReduction(h,
                                          &desc,
                                          dA,
                                          A.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          dC,
                                          D.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          dD,
                                          D.modes.data(),
                                          opOf(op),
                                          computeOf(compute));
        }
        if(st == HIPTENSOR_STATUS_SUCCESS)
        {
            std::string tag = std::to_string(id);
            setenv("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", tag.c_str(), 1);
            hiptensorPlanPreference_t pref = nullptr;
            hiptensorCreatePlanPreference(h,
                                          &pref,
                                          defaultRun ? HIPTENSOR_ALGO_DEFAULT
                                                     : HIPTENSOR_ALGO_DEFAULT_PATIENT,
                                          HIPTENSOR_JIT_MODE_NONE);
            hiptensorCacheMode_t none = HIPTENSOR_CACHE_MODE_NONE;
            hiptensorPlanPreferenceSetAttribute(
                h, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &none, sizeof(none));
            uint64_t ws          = 0;
            st                   = hiptensorEstimateWorkspaceSize(h,
                                                desc,
                                                pref,
                                                defaultRun ? HIPTENSOR_WORKSPACE_DEFAULT
                                                                             : HIPTENSOR_WORKSPACE_MAX,
                                                &ws);
            hiptensorPlan_t plan = nullptr;
            lastSelection.clear();
            double planUs = 0;
            if(st == HIPTENSOR_STATUS_SUCCESS)
            {
                auto p0 = std::chrono::steady_clock::now();
                st      = hiptensorCreatePlan(h, &plan, desc, pref, ws);
                planUs  = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()
                                                                   - p0)
                             .count();
            }
            if(plan != nullptr)
                hiptensorDestroyPlan(plan);
            hiptensorDestroyPlanPreference(pref);
            if(defaultRun)
                std::printf("SEL %ld plan_us=%.1f %s\n", id, planUs, lastSelection.c_str());
        }
        if(desc != nullptr)
            hiptensorDestroyOperationDescriptor(desc);
        for(auto d : descs)
            if(d != nullptr)
                hiptensorDestroyTensorDescriptor(d);
        if(st != HIPTENSOR_STATUS_SUCCESS)
        {
            failed++;
            std::printf("FAIL %ld %s\n", id, hiptensorGetErrorString(st));
        }
        done++;
        if(!progress.empty())
        {
            FILE* f = std::fopen(progress.c_str(), "w");
            if(f)
            {
                std::fprintf(f, "%ld\n", index + 1);
                std::fclose(f);
            }
        }
        std::fflush(stdout);
    }
    std::printf("SWEEP_END done=%ld failed=%ld last=%ld\n", done, failed, index);
    hiptensorLoggerSetCallback(nullptr);
    hiptensorDestroy(h);
    return 0;
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblaslt-bench-partitioned: benchmark hipblaslt_ext::PartitionedGemm, one GEMM split over the
// devices of a partitioned GPU (for example the two partitions of a DPX card).
//
// Every partition is one HIP device (0..partitions-1) with its own handle, stream and workspace.
// The time reported is the full span: the first launch to the last partition done.

#include "program_options.hpp"
#include "utility.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace hipblaslt_ext;
using namespace roc; // for the emulated program_options

namespace
{
    struct Options
    {
        int64_t     m = 4096, n = 4096, k = 4096;
        std::string transA = "T", transB = "N";
        std::string precision  = "f16";
        std::string splitDim   = "N";
        std::string placementA = "replicated", placementB = "replicated", placementC = "replicated",
                    placementD = "replicated";
        int  partitions        = 2;
        int  iters = 20, coldIters = 5;
        int  workspaceMB = 64;
        bool verify = false, single = false;
    };

    // One device allocation, freed on the device that owns it.
    class Buf
    {
    public:
        Buf(int device, size_t bytes, const void* init = nullptr)
            : m_device(device)
        {
            check(hipSetDevice(device), "hipSetDevice");
            check(hipMalloc(&m_ptr, bytes), "hipMalloc");
            if(init != nullptr)
                check(hipMemcpy(m_ptr, init, bytes, hipMemcpyHostToDevice), "hipMemcpy");
            else
                check(hipMemset(m_ptr, 0, bytes), "hipMemset");
        }
        ~Buf()
        {
            if(m_ptr != nullptr)
            {
                static_cast<void>(hipSetDevice(m_device));
                static_cast<void>(hipFree(m_ptr));
            }
        }
        Buf(const Buf&)             = delete;
        Buf&  operator=(const Buf&) = delete;
        void* get() const
        {
            return m_ptr;
        }

        static void check(hipError_t e, const char* what)
        {
            if(e != hipSuccess)
                throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
        }

    private:
        int   m_device;
        void* m_ptr = nullptr;
    };

    void check(hipblasStatus_t s, const char* what)
    {
        if(s != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " returned status "
                                     + std::to_string(int(s)));
    }

    Placement parsePlacement(std::string const& s)
    {
        if(s == "shared")
            return Placement::Shared;
        if(s == "replicated")
            return Placement::Replicated;
        if(s == "sliced")
            return Placement::Sliced;
        throw std::invalid_argument("placement must be shared, replicated or sliced: " + s);
    }

    template <typename T>
    std::vector<T> randomData(size_t count, unsigned seed)
    {
        uint32_t       state = seed * 2654435761u + 1;
        std::vector<T> v(count);
        for(auto& x : v)
        {
            state = state * 1664525u + 1013904223u;
            x     = T(float(state >> 9) / float(1u << 23) - 0.5f);
        }
        return v;
    }

    template <typename T>
    int run(Options const& o, hipDataType type)
    {
        int deviceCount = 0;
        Buf::check(hipGetDeviceCount(&deviceCount), "hipGetDeviceCount");
        if(o.partitions < 1 || o.partitions > deviceCount)
        {
            hipblaslt_cerr << "error: --partitions " << o.partitions << " but " << deviceCount
                           << " device(s) are visible\n";
            return 1;
        }
        const int       P   = o.partitions;
        const bool      tA  = o.transA == "T" || o.transA == "t";
        const bool      tB  = o.transB == "T" || o.transB == "t";
        const auto      opA = tA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const auto      opB = tB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const SplitDim  dim = o.splitDim == "M" || o.splitDim == "m" ? SplitDim::M : SplitDim::N;
        const Placement pA = parsePlacement(o.placementA), pB = parsePlacement(o.placementB),
                        pC = parsePlacement(o.placementC), pD = parsePlacement(o.placementD);
        const int64_t m = o.m, n = o.n, k = o.k;
        // column-major storage and leading dimensions of the full matrices
        const int64_t lda = tA ? k : m, ldb = tB ? n : k, ldc = m, ldd = m;
        const int64_t aCols = tA ? m : k, bCols = tB ? k : n;

        // A is split by M and B by N; a slice must be contiguous columns to be Sliced.
        if(pA == Placement::Sliced && !(dim == SplitDim::M && tA))
            throw std::invalid_argument("--placement_a sliced needs --split_dim M and --transA T");
        if(pB == Placement::Sliced && !(dim == SplitDim::N && !tB))
            throw std::invalid_argument("--placement_b sliced needs --split_dim N and --transB N");
        if(pC == Placement::Sliced)
            throw std::invalid_argument("--placement_c sliced is not supported by this benchmark");

        // slice rule of the library: first P-1 take size/P, the last takes the rest
        const int64_t        total = dim == SplitDim::M ? m : n;
        std::vector<int64_t> off(P), sz(P);
        for(int i = 0; i < P; i++)
        {
            off[i] = i * (total / P);
            sz[i]  = i + 1 < P ? total / P : total - off[i];
        }

        const size_t workspaceBytes = size_t(o.workspaceMB) << 20;
        std::vector<std::unique_ptr<hipblaslt_local_handle>> handleOwners;
        std::vector<hipblasLtHandle_t>                       handles;
        std::vector<hipStream_t>                             streams(P);
        std::vector<std::unique_ptr<Buf>>                    workspaceBufs;
        std::vector<void*>                                   workspaces;
        for(int i = 0; i < P; i++)
        {
            Buf::check(hipSetDevice(i), "hipSetDevice");
            handleOwners.push_back(std::make_unique<hipblaslt_local_handle>());
            handles.push_back(*handleOwners.back());
            Buf::check(hipStreamCreate(&streams[i]), "hipStreamCreate");
            workspaceBufs.push_back(std::make_unique<Buf>(i, workspaceBytes));
            workspaces.push_back(workspaceBufs.back()->get());
        }

        auto           hostA = randomData<T>(size_t(lda) * aCols, 1);
        auto           hostB = randomData<T>(size_t(ldb) * bCols, 2);
        std::vector<T> hostC(size_t(ldc) * n, T(0));

        std::vector<std::unique_ptr<Buf>> bufs;
        auto                              keep = [&](std::unique_ptr<Buf> b) {
            bufs.push_back(std::move(b));
            return bufs.back()->get();
        };
        // peer access to device 0 for Shared matrices
        if(pA == Placement::Shared || pB == Placement::Shared || pC == Placement::Shared
           || pD == Placement::Shared)
            for(int i = 1; i < P; i++)
            {
                Buf::check(hipSetDevice(i), "hipSetDevice");
                hipError_t e = hipDeviceEnablePeerAccess(0, 0);
                if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                    throw std::runtime_error("no peer access from device " + std::to_string(i)
                                             + " to device 0");
                static_cast<void>(hipGetLastError());
            }

        auto makeInput = [&](Placement             p,
                             std::vector<T> const& host,
                             int64_t               ld,
                             int64_t               cols,
                             bool                  sliceColumns) {
            std::vector<const void*> out;
            if(p == Placement::Shared)
                out.push_back(keep(std::make_unique<Buf>(0, host.size() * sizeof(T), host.data())));
            else
                for(int i = 0; i < P; i++)
                {
                    if(p == Placement::Replicated)
                        out.push_back(
                            keep(std::make_unique<Buf>(i, host.size() * sizeof(T), host.data())));
                    else
                    {
                        (void)cols;
                        (void)sliceColumns;
                        std::vector<T> s(host.begin() + off[i] * ld,
                                         host.begin() + (off[i] + sz[i]) * ld);
                        out.push_back(
                            keep(std::make_unique<Buf>(i, s.size() * sizeof(T), s.data())));
                    }
                }
            return out;
        };
        PartitionedGemmInputs in;
        in.a = makeInput(pA, hostA, lda, aCols, true);
        in.b = makeInput(pB, hostB, ldb, bCols, true);
        in.c = makeInput(pC, hostC, ldc, n, false);
        // D: Shared = one buffer, Replicated = a full buffer each, Sliced = just the slice
        const int64_t lddUse
            = pD == Placement::Sliced
                  ? (dim == SplitDim::M ? *std::max_element(sz.begin(), sz.end()) : m)
                  : ldd;
        if(pD == Placement::Shared)
            in.d.push_back(keep(std::make_unique<Buf>(0, size_t(ldd) * n * sizeof(T))));
        else
            for(int i = 0; i < P; i++)
            {
                size_t elems = pD == Placement::Sliced
                                   ? size_t(lddUse) * (dim == SplitDim::M ? n : sz[i])
                                   : size_t(ldd) * n;
                in.d.push_back(keep(std::make_unique<Buf>(i, elems * sizeof(T))));
            }
        float alpha = 1.f, beta = 0.f;
        in.alpha = &alpha;
        in.beta  = &beta;

        PartitionedGemm gemm(handles, opA, opB, type, type, type, type, HIPBLAS_COMPUTE_32F);
        check(
            gemm.setProblem(m, n, k, 1, lda, ldb, ldc, lddUse, 0, 0, 0, 0, dim, pA, pB, pC, pD, in),
            "setProblem");
        gemm.setMaxWorkspaceBytes(workspaceBytes);
        GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspaceBytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
        check(gemm.algoGetHeuristic(1, pref, heuristic), "algoGetHeuristic");
        if(heuristic.empty())
        {
            hipblaslt_cout << "skipped: no algorithm that every partition accepts\n";
            return 0;
        }
        check(gemm.initialize(heuristic[0].algo, workspaces, true, streams), "initialize");

        auto syncAll = [&] {
            for(int i = 0; i < P; i++)
            {
                Buf::check(hipSetDevice(i), "hipSetDevice");
                Buf::check(hipStreamSynchronize(streams[i]), "hipStreamSynchronize");
            }
        };
        for(int i = 0; i < o.coldIters; i++)
            check(gemm.run(streams), "run");
        syncAll();
        const auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < o.iters; i++)
            check(gemm.run(streams), "run");
        syncAll();
        const double us
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                  .count()
              / o.iters;

        // ---- the same GEMM on device 0 (reference values, optional timing) ----
        double         singleUs = 0;
        std::vector<T> ref;
        bool           verified = false, pass = true;
        if(o.verify || o.single)
        {
            Buf a(0, hostA.size() * sizeof(T), hostA.data()),
                b(0, hostB.size() * sizeof(T), hostB.data()),
                c(0, hostC.size() * sizeof(T), hostC.data()), d(0, size_t(ldd) * n * sizeof(T)),
                ws(0, workspaceBytes);
            Gemm            full(handles[0], opA, opB, type, type, type, type, HIPBLAS_COMPUTE_32F);
            GemmEpilogue    epilogue;
            GemmInputs      inputs;
            GemmProblemType problemType(opA, opB, type, type, type, type, HIPBLAS_COMPUTE_32F);
            inputs.setA(a.get());
            inputs.setB(b.get());
            inputs.setC(c.get());
            inputs.setD(d.get());
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            check(full.setProblem(
                      m, n, k, 1, lda, ldb, ldc, ldd, 0, 0, 0, 0, epilogue, inputs, problemType),
                  "reference setProblem");
            full.setMaxWorkspaceBytes(workspaceBytes);
            std::vector<hipblasLtMatmulHeuristicResult_t> fullHeuristic;
            check(full.algoGetHeuristic(1, pref, fullHeuristic), "reference heuristic");
            if(!fullHeuristic.empty())
            {
                Buf::check(hipSetDevice(0), "hipSetDevice");
                check(full.initialize(fullHeuristic[0].algo, ws.get(), true, streams[0]),
                      "reference initialize");
                for(int i = 0; i < o.coldIters; i++)
                    check(full.run(streams[0]), "reference run");
                Buf::check(hipStreamSynchronize(streams[0]), "sync");
                const auto s0 = std::chrono::steady_clock::now();
                for(int i = 0; i < o.iters; i++)
                    check(full.run(streams[0]), "reference run");
                Buf::check(hipStreamSynchronize(streams[0]), "sync");
                singleUs = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - s0)
                               .count()
                           / o.iters;
                ref.resize(size_t(ldd) * n);
                Buf::check(
                    hipMemcpy(ref.data(), d.get(), ref.size() * sizeof(T), hipMemcpyDeviceToHost),
                    "memcpy");
            }
            if(o.verify && !ref.empty())
            {
                verified = true;
                for(int i = 0; i < P && pass; i++)
                {
                    const int64_t  rows = pD == Placement::Sliced ? lddUse : ldd;
                    std::vector<T> got(
                        size_t(rows) * (pD == Placement::Sliced && dim == SplitDim::N ? sz[i] : n));
                    Buf::check(hipSetDevice(pD == Placement::Shared ? 0 : i), "hipSetDevice");
                    Buf::check(hipMemcpy(got.data(),
                                         in.d[pD == Placement::Shared ? 0 : i],
                                         got.size() * sizeof(T),
                                         hipMemcpyDeviceToHost),
                               "memcpy");
                    for(int64_t col = 0; col < (dim == SplitDim::N ? sz[i] : n) && pass; col++)
                        for(int64_t row = 0; row < (dim == SplitDim::M ? sz[i] : m); row++)
                        {
                            const int64_t r    = dim == SplitDim::M ? off[i] + row : row;
                            const int64_t c    = dim == SplitDim::N ? off[i] + col : col;
                            const float   want = float(ref[r + c * ldd]);
                            const float   have = pD == Placement::Sliced
                                                     ? float(got[row + col * rows])
                                                     : float(got[r + c * rows]);
                            if(std::fabs(want - have) > 2e-3f * std::max(1.f, std::fabs(want)))
                            {
                                pass = false;
                                break;
                            }
                        }
                }
            }
        }

        const double gflops = 2.0 * double(m) * double(n) * double(k) / 1e9 / (us * 1e-6);
        hipblaslt_cout
            << "partitions,split_dim,M,N,K,transA,transB,type,placement_a,placement_b,placement_c,"
               "placement_d,hipblaslt-Gflops,us,single_device_us,verify\n";
        hipblaslt_cout << P << "," << (dim == SplitDim::M ? "M" : "N") << "," << m << "," << n
                       << "," << k << "," << o.transA << "," << o.transB << "," << o.precision
                       << "," << o.placementA << "," << o.placementB << "," << o.placementC << ","
                       << o.placementD << "," << gflops << "," << us << ","
                       << (o.single || o.verify ? singleUs : 0) << ","
                       << (verified ? (pass ? "PASS" : "FAIL") : "-") << "\n";

        for(int i = 0; i < P; i++)
        {
            static_cast<void>(hipSetDevice(i));
            static_cast<void>(hipStreamDestroy(streams[i]));
        }
        return pass ? 0 : 1;
    }
} // namespace

int main(int argc, char* argv[])
try
{
    Options             o;
    options_description desc("hipblaslt-bench-partitioned command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m", value<int64_t>(&o.m)->default_value(4096), "Rows of op(A), C and D")
        ("sizen,n", value<int64_t>(&o.n)->default_value(4096), "Columns of op(B), C and D")
        ("sizek,k", value<int64_t>(&o.k)->default_value(4096), "Inner dimension")
        ("transA", value<std::string>(&o.transA)->default_value("T"), "N or T")
        ("transB", value<std::string>(&o.transB)->default_value("N"), "N or T")
        ("precision", value<std::string>(&o.precision)->default_value("f16"), "f16 or f32 (inputs and outputs; fp32 compute)")
        ("partitions", value<int>(&o.partitions)->default_value(2), "Number of partitions (devices 0..partitions-1)")
        ("split_dim", value<std::string>(&o.splitDim)->default_value("N"), "M or N")
        ("placement_a", value<std::string>(&o.placementA)->default_value("replicated"), "shared, replicated or sliced")
        ("placement_b", value<std::string>(&o.placementB)->default_value("replicated"), "shared, replicated or sliced")
        ("placement_c", value<std::string>(&o.placementC)->default_value("replicated"), "shared or replicated")
        ("placement_d", value<std::string>(&o.placementD)->default_value("replicated"), "shared, replicated or sliced")
        ("iters,i", value<int>(&o.iters)->default_value(20), "Timed iterations")
        ("cold_iters,j", value<int>(&o.coldIters)->default_value(5), "Warm-up iterations")
        ("workspace_mb", value<int>(&o.workspaceMB)->default_value(64), "Workspace per partition in MiB")
        ("single", value<bool>(&o.single)->default_value(false), "Also time the full GEMM on device 0")
        ("verify,v", value<bool>(&o.verify)->default_value(false), "Compare every slice with the full GEMM on device 0")
        ("help,h", "produces this help message");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    if(vm.count("help"))
    {
        hipblaslt_cout << desc << "\n";
        return 0;
    }

    if(o.precision == "f16")
        return run<_Float16>(o, HIP_R_16F);
    if(o.precision == "f32")
        return run<float>(o, HIP_R_32F);
    hipblaslt_cerr << "error: --precision must be f16 or f32\n";
    return 1;
}
catch(const std::exception& e)
{
    hipblaslt_cerr << "error: " << e.what() << "\n";
    return 1;
}

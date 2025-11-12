#include <iostream>

#include "get_handle.hpp"
#include "random.hpp"
#include <verify.hpp>
#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <gtest/gtest.h>

#include "perf_helper.hpp"
#include <miopen/float_equal.hpp>

#define MAX_TENSOR_ELEM 17
#define PERF_ENABLE 1
#define POW_2 1

struct TensorsConfig
{
    std::vector<std::size_t> aclens;
    std::vector<std::size_t> acstrides;
    std::vector<std::size_t> blens;
    std::vector<std::size_t> bstrides;
};

static inline std::vector<std::size_t> calcStrideSizes(const std::vector<std::size_t>& lens)
{
    std::vector<std::size_t> strideSizeHolder(lens.size());

    if(lens.empty())
        return strideSizeHolder;

    strideSizeHolder.back() = 1;

    // -2 -> -1 from indexation + -1 starting from H. W-stride is always 1
    for(int i = static_cast<int>(lens.size()) - 2; i >= 0; --i)
        strideSizeHolder[i] = strideSizeHolder[i + 1] * lens[i + 1];

    return strideSizeHolder;
}

// helper for skipping tests archs except one selected
static bool selectArch(const std::string& arch)
{
    const auto& dev = get_handle().GetDeviceName();
    return miopen::StartsWith(dev, arch);
}

template <typename T>
std::vector<TensorsConfig> TensorsConfigs()
{
    if constexpr(PERF_ENABLE)
    {
        constexpr size_t maxTotalSizeInBytes = 2ULL * 1024 * 1024 * 1024;
        const size_t maxTotalSize            = maxTotalSizeInBytes / sizeof(T);
        constexpr size_t minTotalSizeInBytes = 4ULL * 1024 * 1024;
        const size_t minTotalSize            = minTotalSizeInBytes / sizeof(T);

        constexpr size_t BperA           = 6;
        constexpr size_t testsCountLimit = 50;

        std::vector<std::array<size_t, 5>> shapes_A;
        shapes_A.reserve(size_t{1} << 15);

        // N,C,D,H,W POW2 test - with limit size cutoff
        if constexpr(POW_2)
        {
            for(size_t N = 1; N && N <= maxTotalSize; N <<= 1)
                for(size_t C = 1; C <= maxTotalSize / N; C <<= 1)
                    for(size_t D = 1; D <= maxTotalSize / (N * C); D <<= 1)
                        for(size_t H = 1; H <= maxTotalSize / (N * C * D); H <<= 1)
                            for(size_t W = 1; W <= maxTotalSize / (N * C * D * H); W <<= 1)
                            {
                                const size_t total = N * C * D * H * W;
                                if(total < minTotalSize)
                                    continue;
                                if(total > maxTotalSize)
                                    break;
                                shapes_A.push_back({N, C, D, H, W});
                            }
        }

        const size_t shapes_A_kept = std::max<size_t>(1, testsCountLimit / BperA);
        const size_t step =
            std::max<size_t>(1, (shapes_A.size() + shapes_A_kept - 1) / shapes_A_kept);

        std::vector<TensorsConfig> configs;
        configs.reserve(std::min(shapes_A.size() / step, shapes_A_kept) * BperA);

        auto mkv = [](const std::array<size_t, 5>& a) {
            return std::vector<size_t>{a[0], a[1], a[2], a[3], a[4]};
        };
        auto pushCfg = [&](const std::vector<size_t>& A, const std::vector<size_t>& B) {
            configs.push_back({A, calcStrideSizes(A), B, calcStrideSizes(B)});
        };

        for(size_t i = 0; i < shapes_A.size() && configs.size() + BperA <= testsCountLimit;
            i += step)
        {
            const auto A = mkv(shapes_A[i]);

            pushCfg(A, {A[0], A[1], A[2], A[3], A[4]});
            pushCfg(A, {A[0], A[1], A[2], A[3], 1});
            pushCfg(A, {A[0], A[1], A[2], 1, 1});
            pushCfg(A, {A[0], A[1], 1, 1, 1});
            pushCfg(A, {A[0], 1, 1, 1, 1});
            pushCfg(A, {1, A[1], A[2], A[3], A[4]});
        }

        std::cout << "[PerfTest] Generated " << configs.size() << " cases in ["
                  << (minTotalSizeInBytes >> 20) << "MB .. " << (maxTotalSizeInBytes >> 30)
                  << "GB], dtype=" << sizeof(T) << "B\n";

        return configs;
    }
    else
    {
        // fast smoke test, borders + overflow
        std::vector<TensorsConfig> v;
        std::vector<std::size_t> a1{16, 8, 4, 4, 2};
        std::vector<std::size_t> a2{32, 16, 8, 4, 2};
        v.push_back({a1, calcStrideSizes(a1), a1, calcStrideSizes(a1)});
        v.push_back({a1, calcStrideSizes(a1), {16, 8, 4, 4, 1}, calcStrideSizes({16, 8, 4, 4, 1})});
        v.push_back({a1, calcStrideSizes(a1), {16, 8, 4, 1, 1}, calcStrideSizes({16, 8, 4, 1, 1})});
        v.push_back({a1, calcStrideSizes(a1), {16, 8, 1, 1, 1}, calcStrideSizes({16, 8, 1, 1, 1})});
        v.push_back({a2, calcStrideSizes(a2), a2, calcStrideSizes(a2)});
        return v;
    }
}

template <typename T>
struct Op5DTensorGenericTest
    : public ::testing::TestWithParam<std::tuple<TensorsConfig, float, float, float>>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        handle.EnableProfiling(true);
        std::tie(tensorsConfig, alpha0, alpha1, beta) = GetParam();

        const std::string testedArchitecture = "gfx1030";
        if(!selectArch(testedArchitecture))
        {
            GTEST_SKIP() << "This test is limited to " << testedArchitecture << " only.";
        }

        data_type = miopen_type<T>{};

        // fill test tensors
        tensA = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensB = tensor<T>{tensorsConfig.blens, tensorsConfig.bstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});
        tensC = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides}.generate(
            tensor_elem_gen_integer{MAX_TENSOR_ELEM});

        // Write the device tensors
        tensA_dev = handle.Write(tensA.data);
        tensB_dev = handle.Write(tensB.data);

        // Allocate output tensors for OCL and HIP
        tensC_ocl = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};
        tensC_hip = tensor<T>{tensorsConfig.aclens, tensorsConfig.acstrides};

        // kernel params preparation
        const int a_n = static_cast<int>(tensorsConfig.aclens[0]);
        const int a_c = static_cast<int>(tensorsConfig.aclens[1]);
        const int a_d = static_cast<int>(tensorsConfig.aclens[2]);
        const int a_h = static_cast<int>(tensorsConfig.aclens[3]);
        const int a_w = static_cast<int>(tensorsConfig.aclens[4]);

        const int b_n = static_cast<int>(tensorsConfig.blens[0]);
        const int b_c = static_cast<int>(tensorsConfig.blens[1]);
        const int b_d = static_cast<int>(tensorsConfig.blens[2]);
        const int b_h = static_cast<int>(tensorsConfig.blens[3]);
        const int b_w = static_cast<int>(tensorsConfig.blens[4]);

        bitmap = 0u;
        if(b_w > 1)
            bitmap |= (1u << 0); // W
        if(b_h > 1)
            bitmap |= (1u << 1); // H
        if(b_d > 1)
            bitmap |= (1u << 2); // D
        if(b_c > 1)
            bitmap |= (1u << 3); // C
        if(b_n > 1)
            bitmap |= (1u << 4); // N

        long long wpw = 1;
        if(b_w == 1)
            wpw *= a_w;
        if(b_h == 1)
            wpw *= a_h;
        if(b_d == 1)
            wpw *= a_d;
        if(b_c == 1)
            wpw *= a_c;
        if(b_n == 1)
            wpw *= a_n;

        work_per_wg =
            static_cast<int64_t>(std::min<long long>(wpw, std::numeric_limits<int64_t>::max()));

        // sanity check
        const long long total_ll = 1ll * a_n * a_c * a_d * a_h * a_w;
        const long long numwg_ll = (b_w > 1 ? b_w : 1) * 1ll * (b_h > 1 ? b_h : 1) *
                                   (b_d > 1 ? b_d : 1) * (b_c > 1 ? b_c : 1) * (b_n > 1 ? b_n : 1);

        if(numwg_ll * wpw != total_ll)
        {
            std::cerr << "Partition mismatch: total=" << total_ll << " num_wg=" << numwg_ll
                      << " work_per_wg=" << wpw << "  A={" << a_n << "," << a_c << "," << a_d << ","
                      << a_h << "," << a_w << "}" << "  B={" << b_n << "," << b_c << "," << b_d
                      << "," << b_h << "," << b_w << "}" << "  bitmap=" << bitmap << std::endl;
        }

        long long num_wg_ll = 1;
        num_wg_ll *= (b_w > 1 ? b_w : 1);
        num_wg_ll *= (b_h > 1 ? b_h : 1);
        num_wg_ll *= (b_d > 1 ? b_d : 1);
        num_wg_ll *= (b_c > 1 ? b_c : 1);
        num_wg_ll *= (b_n > 1 ? b_n : 1);

        num_wg_orig       = static_cast<int>(num_wg_ll);
        max_num_wg        = 4096;
        int num_wg_launch = std::min(num_wg_orig, max_num_wg);

        const size_t local_threads =
            std::min<size_t>(256, static_cast<size_t>(std::max(1, work_per_wg)));

        vld                         = {local_threads, 1, 1};
        const size_t global_threads = static_cast<size_t>(num_wg_launch) * local_threads;
        vgd                         = {global_threads, 1, 1};

        network_config = std::to_string(data_type) + "-miopenTensorOpAdd" + "-gt" +
                         std::to_string(global_threads) + "-lt" + std::to_string(local_threads) +
                         "-bm" + std::to_string(bitmap) + "-wpw" + std::to_string(work_per_wg) +
                         "-wg" + std::to_string(num_wg_orig) + "-MAXWG" +
                         std::to_string(num_wg_launch);
    }

    void runOCL()
    {
        std::cout << "Running OCL:" << std::endl;
        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);
        std::fill(tensC_ocl.begin(), tensC_ocl.end(), std::numeric_limits<T>::quiet_NaN());

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::OpenCL{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_5D_TENSOR_GENERIC";

        std::string program_name       = "MIOpenTensorKernels.cl";
        std::string network_config_ocl = network_config + "-ocl";

        const int b_nstride = (tensorsConfig.blens[0] == 1) ? 0 : (int)tensorsConfig.bstrides[0];
        const int b_cstride = (tensorsConfig.blens[1] == 1) ? 0 : (int)tensorsConfig.bstrides[1];
        const int b_dstride = (tensorsConfig.blens[2] == 1) ? 0 : (int)tensorsConfig.bstrides[2];
        const int b_hstride = (tensorsConfig.blens[3] == 1) ? 0 : (int)tensorsConfig.bstrides[3];

        handle.AddKernel("Op5dTensorGeneric",
                         network_config_ocl,
                         program_name,
                         "Op5dTensorGeneric",
                         vld,
                         vgd,
                         params)(
            // A
            tensA_dev.get(),
            static_cast<int>(tensorsConfig.acstrides[0]),
            static_cast<int>(tensorsConfig.acstrides[1]),
            static_cast<int>(tensorsConfig.acstrides[2]),
            static_cast<int>(tensorsConfig.acstrides[3]),

            // B
            tensB_dev.get(),
            static_cast<int>(tensorsConfig.blens[1]),
            static_cast<int>(tensorsConfig.blens[2]),
            static_cast<int>(tensorsConfig.blens[3]),
            static_cast<int>(tensorsConfig.blens[4]),
            b_nstride,
            b_cstride,
            b_dstride,
            b_hstride,

            // C
            tensC_dev.get(),
            static_cast<int>(tensorsConfig.aclens[1]),
            static_cast<int>(tensorsConfig.aclens[2]),
            static_cast<int>(tensorsConfig.aclens[3]),
            static_cast<int>(tensorsConfig.aclens[4]),
            static_cast<int>(tensorsConfig.acstrides[0]),
            static_cast<int>(tensorsConfig.acstrides[1]),
            static_cast<int>(tensorsConfig.acstrides[2]),
            static_cast<int>(tensorsConfig.acstrides[3]),

            // scalars & misc
            alpha0,
            alpha1,
            beta,
            bitmap,
            work_per_wg,
            static_cast<int64_t>(0),
            static_cast<int64_t>(0),
            static_cast<int64_t>(0),
            static_cast<int>(num_wg_orig));

        handle.Finish();
        tensC_ocl.data = handle.Read<T>(tensC_dev, tensC_ocl.data.size());

        if constexpr(PERF_ENABLE)
        {
            ph.perfTest(handle,
                        "Op5dTensorGeneric",
                        network_config_ocl,
                        false,

                        // A
                        tensA_dev.get(),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        static_cast<int>(tensorsConfig.acstrides[1]),
                        static_cast<int>(tensorsConfig.acstrides[2]),
                        static_cast<int>(tensorsConfig.acstrides[3]),

                        // B
                        tensB_dev.get(),
                        static_cast<int>(tensorsConfig.blens[1]),
                        static_cast<int>(tensorsConfig.blens[2]),
                        static_cast<int>(tensorsConfig.blens[3]),
                        static_cast<int>(tensorsConfig.blens[4]),
                        b_nstride,
                        b_cstride,
                        b_dstride,
                        b_hstride,

                        // C
                        tensC_dev.get(),
                        static_cast<int>(tensorsConfig.aclens[1]),
                        static_cast<int>(tensorsConfig.aclens[2]),
                        static_cast<int>(tensorsConfig.aclens[3]),
                        static_cast<int>(tensorsConfig.aclens[4]),
                        static_cast<int>(tensorsConfig.acstrides[0]),
                        static_cast<int>(tensorsConfig.acstrides[1]),
                        static_cast<int>(tensorsConfig.acstrides[2]),
                        static_cast<int>(tensorsConfig.acstrides[3]),

                        // scalars & misc
                        alpha0,
                        alpha1,
                        beta,
                        bitmap,
                        work_per_wg,
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int64_t>(0),
                        static_cast<int>(num_wg_orig));
        }
    }

    void runHIP()
    {
        std::cout << "Running HIP:" << std::endl;

        auto&& handle = get_handle();
        tensC_dev     = handle.Write(tensC.data);

        std::fill(tensC_hip.begin(), tensC_hip.end(), std::numeric_limits<T>::quiet_NaN());

        params = " -DMIOPEN_TYPE=" + miopen::GetDataType(data_type) +
                 " -DMAX_NUM_WG=" + std::to_string(max_num_wg);
        params += " " + miopen::GetDataTypeKBP(data_type).GenerateFor(miopen::kbp::HIP{});
        params += " -DMIOPEN_TENSOR_OP=miopenAdd";

        auto is_contig = [&](const std::vector<size_t>& lens, const std::vector<size_t>& strides) {
            uint64_t exp = 1;
            for(int i = 4; i >= 0; --i)
            {
                if(static_cast<uint64_t>(strides[i]) != exp)
                    return false;
                exp *= static_cast<uint64_t>(lens[i]);
            }
            return true;
        };

        const bool a_contig = is_contig(tensorsConfig.aclens, tensorsConfig.acstrides);
        const bool b_contig = is_contig(tensorsConfig.blens, tensorsConfig.bstrides);
        const bool c_contig = is_contig(tensorsConfig.aclens, tensorsConfig.acstrides);
        const bool shapes_equal = (tensorsConfig.aclens == tensorsConfig.blens);

        const char* kname;
        if(shapes_equal && a_contig && b_contig && c_contig)
            kname = "Op5dTensorGenericContiguous";
        else
            kname = "Op5dTensorGeneric";

        // нужный feature-флаг под выбранное ядро
        if(std::string{kname} == "Op5dTensorGenericContiguous")
            params += " -DUSE_5D_TENSOR_GENERIC_CONTIGUOUS";
        else
            params += " -DUSE_5D_TENSOR_GENERIC";

        // узкий индекс когда безопасно (ускоряет деления/моды в generic и делает код легче)
        // {
        //     unsigned __int128 total = 1;
        //     for(int i=0;i<5;++i) total *= static_cast<unsigned long long>(tensorsConfig.aclens[i]);
        //     const bool fits32 = (total <= static_cast<unsigned __int128>(0xFFFFFFFFull));
        //     if(fits32) params += " -DUSE_INDEX32";
        // }

        std::string program_name       = "MIOpenTensorKernelsHip.cpp";
        std::string network_config_hip = network_config + "-hip-" + kname;

        // --- offsets ---
        const uint64_t Aoff = 0, Boff = 0, Coff = 0;
        
        const uint64_t b_n        = static_cast<uint64_t>(tensorsConfig.blens[0]);
        const uint64_t b_c        = static_cast<uint64_t>(tensorsConfig.blens[1]);
        const uint64_t b_d        = static_cast<uint64_t>(tensorsConfig.blens[2]);
        const uint64_t b_h        = static_cast<uint64_t>(tensorsConfig.blens[3]);
        const uint64_t b_w        = static_cast<uint64_t>(tensorsConfig.blens[4]);
        const uint64_t c_n        = tensorsConfig.aclens[0];
        const uint64_t c_c        = tensorsConfig.aclens[1];
        const uint64_t c_d        = tensorsConfig.aclens[2];
        const uint64_t c_h        = tensorsConfig.aclens[3];
        const uint64_t c_w        = tensorsConfig.aclens[4];
        const uint64_t total_work = c_n * c_c * c_d * c_h * c_w;
        const bool use_beta       = (beta != static_cast<T>(0));

        const uint64_t b_nstride =
            (tensorsConfig.blens[0] == 1) ? 0ull : (uint64_t)tensorsConfig.bstrides[0];
        const uint64_t b_cstride =
            (tensorsConfig.blens[1] == 1) ? 0ull : (uint64_t)tensorsConfig.bstrides[1];
        const uint64_t b_dstride =
            (tensorsConfig.blens[2] == 1) ? 0ull : (uint64_t)tensorsConfig.bstrides[2];
        const uint64_t b_hstride =
            (tensorsConfig.blens[3] == 1) ? 0ull : (uint64_t)tensorsConfig.bstrides[3];
        const uint64_t b_wstride =
            (tensorsConfig.blens[4] == 1) ? 0ull : (uint64_t)tensorsConfig.bstrides[4];

        // std::cout << "Adding Kernels" << std::endl;
        if(kname == "Op5dTensorGenericContiguous")
        {
            std::cout << "Adding kernel for contiguous case:" << std::endl;
            handle.AddKernel(
                "Op5dTensorGenericContiguous", network_config_hip, program_name, kname, vld, vgd, params)(
                // A, B, C
                tensA_dev.get(),
                tensB_dev.get(),
                tensC_dev.get(),

                // offsets
                Aoff,
                Boff,
                Coff,
                
                b_n,
                b_c,
                b_d,
                b_h,
                b_w,

                // C dims
                c_n,
                c_c,
                c_d,
                c_h,
                c_w,

                // scalars
                alpha0,
                alpha1,
                beta,

                // misc
                total_work,
                use_beta);
        }
        else
        {
            std::cout << "Adding Kernel For generic case" << std::endl;
            handle.AddKernel(
                "Op5dTensorGeneric", network_config_hip, program_name, kname, vld, vgd, params)(
                // A, B, C
                tensA_dev.get(),
                tensB_dev.get(),
                tensC_dev.get(),

                // offsets
                Aoff,
                Boff,
                Coff,

                // (FIX) реальные размеры B
                b_n,
                b_c,
                b_d,
                b_h,
                b_w,

                // C dims
                c_n,
                c_c,
                c_d,
                c_h,
                c_w,

                // A strides (5)
                (uint64_t)tensorsConfig.acstrides[0],
                (uint64_t)tensorsConfig.acstrides[1],
                (uint64_t)tensorsConfig.acstrides[2],
                (uint64_t)tensorsConfig.acstrides[3],
                (uint64_t)tensorsConfig.acstrides[4],

                // B strides (5)
                b_nstride,
                b_cstride,
                b_dstride,
                b_hstride,
                b_wstride,

                // C strides (5) — C==A
                (uint64_t)tensorsConfig.acstrides[0],
                (uint64_t)tensorsConfig.acstrides[1],
                (uint64_t)tensorsConfig.acstrides[2],
                (uint64_t)tensorsConfig.acstrides[3],
                (uint64_t)tensorsConfig.acstrides[4],

                // scalars
                alpha0,
                alpha1,
                beta,

                // misc
                total_work,
                use_beta);
        }

        handle.Finish();
        tensC_hip.data = handle.Read<T>(tensC_dev, tensC_hip.data.size());

        if constexpr(PERF_ENABLE)
        {
            std::cout << "Performance test for " << kname << " case" << std::endl;

            if(kname == "Op5dTensorGenericContiguous")
            {
                ph.perfTest(handle,
                            "Op5dTensorGenericContiguous",
                            network_config_hip,
                            false,
                            tensA_dev.get(),
                            tensB_dev.get(),
                            tensC_dev.get(),
                            Aoff,
                            Boff,
                            Coff,
                            b_n,
                            b_c,
                            b_d,
                            b_h,
                            b_w,
                            c_n,
                            c_c,
                            c_d,
                            c_h,
                            c_w,
                            alpha0,
                            alpha1,
                            beta,
                            total_work,
                            use_beta);
            }
            else
            {
                ph.perfTest(handle,
                            "Op5dTensorGeneric",
                            network_config_hip,
                            false,

                            // A, B, C
                            tensA_dev.get(),
                            tensB_dev.get(),
                            tensC_dev.get(),

                            // offsets
                            Aoff,
                            Boff,
                            Coff,

                            // (FIX) реальные размеры B
                            b_n,
                            b_c,
                            b_d,
                            b_h,
                            b_w,

                            // C dims
                            c_n,
                            c_c,
                            c_d,
                            c_h,
                            c_w,

                            // A strides (5)
                            (uint64_t)tensorsConfig.acstrides[0],
                            (uint64_t)tensorsConfig.acstrides[1],
                            (uint64_t)tensorsConfig.acstrides[2],
                            (uint64_t)tensorsConfig.acstrides[3],
                            (uint64_t)tensorsConfig.acstrides[4],

                            // B strides (5)
                            b_nstride,
                            b_cstride,
                            b_dstride,
                            b_hstride,
                            b_wstride,

                            // C strides (5) — C==A
                            (uint64_t)tensorsConfig.acstrides[0],
                            (uint64_t)tensorsConfig.acstrides[1],
                            (uint64_t)tensorsConfig.acstrides[2],
                            (uint64_t)tensorsConfig.acstrides[3],
                            (uint64_t)tensorsConfig.acstrides[4],

                            // scalars
                            alpha0,
                            alpha1,
                            beta,

                            // misc
                            total_work,
                            use_beta);
            }
        }
    }

    void verify()
    {
        ASSERT_EQ(tensC_ocl.data.size(), tensC_hip.data.size());

        const double err = miopen::rms_range(tensC_ocl, tensC_hip);

        // scale = RMS OCL-результата (референс)
        long double ss = 0.0L;
        for(const auto& v : tensC_ocl)
            ss += static_cast<long double>(v) * v;
        const double scale =
            std::sqrt(static_cast<double>(ss) / std::max<size_t>(tensC_ocl.data.size(), 1));

        const double rel_err = (scale > 0.0) ? (err / scale) : err;

        // FP32: простой порог
        constexpr double rtol = 5e-4; // при необходимости можно ослабить до 5e-4
        EXPECT_LT(rel_err, rtol) << "Relative RMS error " << rel_err << " exceeds tolerance "
                                 << rtol;
    }

    void TearDown() override
    {
        if constexpr(PERF_ENABLE)
        {
            auto cat5 = [](const std::vector<std::size_t>& v) {
                return std::to_string(v[0]) + "_" + std::to_string(v[1]) + "_" +
                       std::to_string(v[2]) + "_" + std::to_string(v[3]) + "_" +
                       std::to_string(v[4]);
            };
            std::string stats{};
            stats += "_aclens_" + cat5(tensorsConfig.aclens) + "_acstrides_" +
                     cat5(tensorsConfig.acstrides);
            stats +=
                "_blens_" + cat5(tensorsConfig.blens) + "_bstrides_" + cat5(tensorsConfig.bstrides);
            stats += "_alpha0_" + std::to_string(alpha0) + "_alpha1_" + std::to_string(alpha1) +
                     "_beta_" + std::to_string(beta) + "_" + miopen::GetDataType(data_type);
            ph.writeStatsToCSV("tensor_5d.csv", stats);
        }
    }

    std::string network_config{};
    std::string params{};
    std::vector<size_t> vld, vgd;
    unsigned int bitmap;
    int work_per_wg;
    int num_wg_orig;
    int max_num_wg;

    tensor<T> tensA;
    tensor<T> tensB;
    tensor<T> tensC;
    tensor<T> tensC_ocl;
    tensor<T> tensC_hip;

    miopenDataType_t data_type;

    miopen::Allocator::ManageDataPtr tensA_dev;
    miopen::Allocator::ManageDataPtr tensB_dev;
    miopen::Allocator::ManageDataPtr tensC_dev;

    TensorsConfig tensorsConfig;
    float alpha0, alpha1, beta;

    PerfHelper<T> ph;
};

struct GPU_Op5dTensorGenericTest_FP32 : Op5DTensorGenericTest<float>
{
};

TEST_P(GPU_Op5dTensorGenericTest_FP32, PortTest)
{
    runOCL();
    runHIP();
    verify();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Op5dTensorGenericTest_FP32,
                         testing::Combine(testing::ValuesIn(TensorsConfigs<float>()),
                                          testing::Values(1.0f),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f, 1.0f)));

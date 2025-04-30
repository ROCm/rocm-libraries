/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "kernel_launcher.hpp"
#include "activation.hpp"
#include "definitions.h"
#include "handle.h"
#include "hip_solution_adapter.hpp"
#include "hipsparselt_ostream.hpp"
#include "rocsparselt-types.h"
#include "rocsparselt.h"
#include "status.h"
#include "utility.hpp"

#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <unistd.h>

#define ROCSPARSELT_LIB_PATH "/opt/rocm/hipsparselt/lib"

namespace
{
#ifndef WIN32
    std::string rocsparselt_so_path;

    int rocsparselt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        // uncomment to see all dependent .so files
        //fprintf(stderr, "rocsparselt so file: %s\n", hdr_info->dlpi_name);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "hipsparselt."))
        {
            rocsparselt_so_path = hdr_info->dlpi_name;
        }
        return 0;
    }
#endif

    size_t totalAllcoatedElement(std::vector<size_t>& sizes,
                                 std::vector<size_t>& strides,
                                 size_t               offset)
    {

        size_t totalAllocatedElements = 1;
        for(int i = 0; i < sizes.size(); i++)
            totalAllocatedElements += strides[i] * (sizes[i] - 1);
        totalAllocatedElements += offset;
        return totalAllocatedElements;
    }

    size_t totalAllcoatedElementNonBatch(std::vector<size_t>& sizes,
                                         std::vector<size_t>& strides,
                                         BatchIndices&        batchIndex)
    {
        size_t totalAllocatedElementsNonBatch = 1;
        for(int idx = 0; idx < sizes.size(); idx++)
        {
            bool isBatch = batchIndex.end()
                           != std::find_if(batchIndex.begin(),
                                           batchIndex.end(),
                                           [idx](const BatchIndex& bi) { return bi.a == idx; });
            if(!isBatch)
                totalAllocatedElementsNonBatch += strides[idx] * (sizes[idx] - 1);
        }
        return totalAllocatedElementsNonBatch;
    };

    template <typename Ti, typename To, typename Tc>
    auto ConstructKernelInvoke(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                               const KernelParams&                              kernel)
    {
        KernelInvocation ki;

        ki.args = KernelArguments();

        ki.args.reserve(1024, 128);

        ki.kernelName = kernel.SolutionNameMin;

        ki.workGroupSize.x = kernel.WorkGroup[0] * kernel.WorkGroup[1] * kernel.WorkGroup[2];
        ki.workGroupSize.y = 1;
        ki.workGroupSize.z = 1;

        ki.numWorkGroups.x = 1;
        ki.numWorkGroups.y = 1;

        // Indices for contraction problem
        FreeIndices  freeIndex(2);
        BoundIndices boundIndex(1);
        BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        auto k  = prob.k && *prob.alpha ? prob.k : 0;
        auto ck = prob.sparseA ? k / 2 : k;

        std::vector<size_t> sizes_a(3), sizes_b(3), sizes_c(3), sizes_d(3);
        std::vector<size_t> strides_a = {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a};
        std::vector<size_t> strides_b = {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b};
        std::vector<size_t> strides_c = {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c};
        std::vector<size_t> strides_d = {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d};

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocsparselt_operation_none)
        {
            sizes_a[0] = ck;
            sizes_a[1] = prob.m;
            sizes_a[2] = prob.batch_count;

            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            sizes_a[0] = prob.m;
            sizes_a[1] = ck;
            sizes_a[2] = prob.batch_count;

            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocsparselt_operation_none)
        {
            sizes_b[0] = prob.n;
            sizes_b[1] = k;
            sizes_b[2] = prob.batch_count;

            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            sizes_b[0] = k;
            sizes_b[1] = prob.n;
            sizes_b[2] = prob.batch_count;

            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        sizes_c[0] = prob.m;
        sizes_c[1] = prob.n;
        sizes_c[2] = prob.batch_count;

        sizes_d[0] = prob.m;
        sizes_d[1] = prob.n;
        sizes_d[2] = prob.batch_count;

        FreeIndices         freeIndicesA;
        FreeIndices         freeIndicesB;
        std::vector<size_t> freeSizesA;
        std::vector<size_t> freeSizesB;

        freeIndicesA.reserve(freeIndex.size());
        freeIndicesB.reserve(freeIndex.size());
        freeSizesA.reserve(freeIndex.size());
        freeSizesB.reserve(freeIndex.size());
        for(int i = 0; i < freeIndex.size(); i++)
        {
            size_t mySize = sizes_d[freeIndex[i].d];
            if(freeIndex[i].isA)
            {
                freeIndicesA.push_back(freeIndex[i]);
                freeSizesA.push_back(mySize);
            }
            else
            {
                freeIndicesB.push_back(freeIndex[i]);
                freeSizesB.push_back(mySize);
            }
        }

        for(size_t i = 0; i < freeIndicesA.size(); i++)
        {
            ki.numWorkGroups.x *= freeSizesA[i];
        }
        for(size_t i = 0; i < freeIndicesB.size(); i++)
        {
            ki.numWorkGroups.y *= freeSizesB[i];
        }

        ki.numWorkGroups.z = 1;

        std::vector<size_t> batchSizes(batchIndex.size());
        std::vector<size_t> boundSizes(boundIndex.size());
        for(int i = 0; i < batchIndex.size(); i++)
        {
            batchSizes[i] = std::max({sizes_a[batchIndex[i].a],
                                      sizes_b[batchIndex[i].b],
                                      sizes_c.empty() ? 0 : sizes_c[batchIndex[i].c],
                                      sizes_d[batchIndex[i].d]});
        }

        for(int i = 0; i < boundIndex.size(); i++)
        {
            boundSizes[i] = std::max(sizes_a[boundIndex[i].a], sizes_b[boundIndex[i].b]);
        }

        for(size_t i = 0; i < batchIndex.size(); i++)
        {
            if(kernel.PackBatchDims & 0x1)
                ki.numWorkGroups.x *= batchSizes[i];
            if(kernel.PackBatchDims & 0x2)
                ki.numWorkGroups.y *= batchSizes[i];
            if(!kernel.PackBatchDims)
                ki.numWorkGroups.z *= batchSizes[i];
        }

        // CD always contain index0.  if this is in the B free indices, then need to
        // transposing the output tensor.
        bool transposeC01 = freeIndicesB.end()
                            != std::find_if(freeIndicesB.begin(),
                                            freeIndicesB.end(),
                                            [](const FreeIndex& fi) { return fi.c == 0 /*idx0*/; });

        if(transposeC01)
            std::swap(ki.numWorkGroups.x, ki.numWorkGroups.y);

        ki.numWorkGroups.x = CeilDivide(ki.numWorkGroups.x, kernel.MacroTile[0]);
        ki.numWorkGroups.y = CeilDivide(ki.numWorkGroups.y, kernel.MacroTile[1]);

        uint32_t problemNumGroupTiles0 = ki.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = ki.numWorkGroups.y;

        ki.numWorkGroups.y *= kernel.GlobalSplitU;

        ki.numWorkItems.x = ki.workGroupSize.x * ki.numWorkGroups.x;
        ki.numWorkItems.y = ki.workGroupSize.y * ki.numWorkGroups.y;
        ki.numWorkItems.z = ki.workGroupSize.z * ki.numWorkGroups.z;

        ki.sharedMemBytes = 0;

        uint64_t tensor2dSizeC = totalAllcoatedElement(sizes_c, strides_c, (size_t)0);
        uint64_t tensor2dSizeA
            = (kernel.PackBatchDims & 0x1)
                  ? totalAllcoatedElement(sizes_a, strides_a, (size_t)0)
                  : totalAllcoatedElementNonBatch(sizes_a, strides_a, batchIndex);
        uint64_t tensor2dSizeB
            = (kernel.PackBatchDims & 0x2)
                  ? totalAllcoatedElement(sizes_b, strides_b, (size_t)0)
                  : totalAllcoatedElementNonBatch(sizes_b, strides_b, batchIndex);

        ki.args.append<uint64_t>("tensor2dSizeC", tensor2dSizeC);
        ki.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
        ki.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);

        ki.args.append<To const*>("d", prob.D);
        ki.args.append<To const*>("c", prob.C);
        ki.args.append<Ti const*>("a", prob.A);
        ki.args.append<Ti const*>("b", prob.B);

        if(prob.sparseA)
            ki.args.append<unsigned char const*>("metadata", prob.metadata);

        ki.args.append<float>("alpha", *prob.alpha);
        ki.args.append<float>("beta", *prob.beta);

        hipsparselt_activation_type act_type
            = string_to_hipsparselt_activation_type(kernel.ActivationType);
        if((act_type != hipsparselt_activation_type::none) && kernel.ActivationFused
           && (!kernel.GlobalAccumulation))
        {
            if(kernel.ActivationHPA)
            {
                //same as the alpha/beta type.
                ki.args.append<float>("activation_0", prob.act_arg0);
                ki.args.append<float>("activation_1", prob.act_arg1);
            }
            else
            {
                ki.args.append<To>("activation_0", static_cast<To>(prob.act_arg0));
                ki.args.append<To>("activation_1", static_cast<To>(prob.act_arg1));
            }
            ki.args.append<uint32_t>("activationType", static_cast<uint32_t>(prob.act_type));
        }

        size_t startStrideCD = kernel.UseInitialStridesCD ? 0 : 1;
        size_t startStrideAB = kernel.UseInitialStridesAB ? 0 : 1;

        for(size_t i = startStrideCD; i < sizes_d.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideD", i), strides_d[i]);

        for(size_t i = startStrideCD; i < sizes_c.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideC", i), strides_c[i]);

        for(size_t i = startStrideAB; i < sizes_a.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideA", i), strides_a[i]);

        for(size_t i = startStrideAB; i < sizes_b.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideB", i), strides_b[i]);

        std::vector<size_t> problemSizes;
        problemSizes.resize(0);
        problemSizes.reserve(sizes_c.size() + boundSizes.size());
        problemSizes.insert(problemSizes.end(), sizes_c.begin(), sizes_c.end());
        problemSizes.insert(problemSizes.end(), boundSizes.begin(), boundSizes.end());

        int idx = 0;
        for(auto size : problemSizes)
        {
            ki.args.append<uint32_t>(concatenate_if<true>("size_", idx), size);
            idx++;
        }

        // Caculate staggerU
        uint32_t sizeL = boundSizes[0];

        // how many stride-sized clicks to stagger start offset
        unsigned int staggerUIter = kernel.StaggerU;

        // /DepthU/GSU
        int unrollLoopIters = sizeL / kernel.DepthU / kernel.GlobalSplitU;

        unsigned int shifted = 1 << kernel.StaggerStrideShift;

        while(staggerUIter > 1)
        {
            if(unrollLoopIters >= (staggerUIter * shifted))
                break;

            staggerUIter /= 2; // step down to smaller stagger
        }

        if(staggerUIter >= 1)
            staggerUIter -= 1;

        ki.args.append<int32_t>("staggerUIter", staggerUIter);
        ki.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        ki.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);

        uint32_t numFullBlocks            = problemNumGroupTiles1;
        uint32_t wgmRemainder1            = 0;
        uint32_t magicNumberWgmRemainder1 = 0;

        if(kernel.WorkGroupMapping != 0)
        {
            numFullBlocks = problemNumGroupTiles1 / kernel.WorkGroupMapping;
            wgmRemainder1 = problemNumGroupTiles1 % kernel.WorkGroupMapping;
            if(wgmRemainder1 == 0)
                wgmRemainder1 = kernel.WorkGroupMapping;

            uint64_t  magicNum;
            const int smallMagicShift = 31;
            magicNum                  = (1L << smallMagicShift) / wgmRemainder1 + 1;
            assert(magicNum >> 32 == 0); // ensure magic number fits
            magicNumberWgmRemainder1 = static_cast<uint32_t>(magicNum);
        }

        ki.args.append<uint32_t>("numFullBlocks", numFullBlocks);
        ki.args.append<uint32_t>("wgmRemainder1", wgmRemainder1);
        ki.args.append<uint32_t>("magicNumberWgmRemainder1", magicNumberWgmRemainder1);

        ki.args.append<uint32_t>("offsetD", prob.buffer_offset_b);
        ki.args.append<uint32_t>("offsetC", prob.buffer_offset_c);
        ki.args.append<uint32_t>("offsetA", prob.buffer_offset_a);
        ki.args.append<uint32_t>("offsetB", prob.buffer_offset_b);

        ki.args.append<uint32_t>("pad", 0);
        return ki;
    }

    /**************************************************
     * The KernelLauncher struct interfaces           *
     **************************************************/
    class KernelLauncher
    {
        std::shared_ptr<hipDeviceProp_t> m_deviceProp;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                    mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        KernelLauncher()
            : m_adapters(GetDeviceCount())
        {
            // We mark KernelLauncher as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocsparselt_internal_kl_is_initialized() = true;
        }

        // KernelLauncher is not copyable or assignable
        KernelLauncher(const KernelLauncher&) = delete;
        KernelLauncher& operator=(const KernelLauncher&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                hipsparselt_cerr << "\nrocsparselt error: Could not initialize Kernel Launcher "
                                    "host: No devices found"
                                 << std::endl;
                hipsparselt_abort();
            }
            return count;
        }

        ~KernelLauncher()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_device_property() const
        {
            return m_deviceProp;
        }

        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
         * Testpath() tests that a path exists and is readable *
         *******************************************************/
        static bool TestPath(const std::string& path)
        {
#ifdef WIN32
            return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
            return access(path.c_str(), R_OK) == 0;
#endif
        }

        /*********************************************************************
         * Initialize adapter and library according to environment variables *
         * and default paths based on librocsparselt.so location and GPU         *
         *********************************************************************/
        void initialize(SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocsparselt_internal_get_arch_name();

            const char* env = getenv("ROCSPARSELT_SPMM_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                path = ROCSPARSELT_LIB_PATH;

                // Find the location of librocsparselt.so
                // Fall back on hard-coded path if static library or not found

#ifndef ROCSPARSELT_STATIC_LIB
                dl_iterate_phdr(rocsparselt_dl_iterate_phdr_callback, NULL);
                if(rocsparselt_so_path.size())
                    path = std::string{dirname(&rocsparselt_so_path[0])};
#endif // ifndef ROCSPARSELT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/../SPMM_KERNELS/library"))
                    path += "/../SPMM_KERNELS/library";
                else if(TestPath(path + "/library"))
                    path += "/library";
                else
                    path += "/hipsparselt/library";
            }

            auto dir      = path + "/libspmm_kernels_" + processor + ".so";
            bool no_match = false;
            if(TestPath(dir))
            {
                if(adapter.loadLibrary(dir) != hipSuccess)
                    no_match = true;
            }
            else
                no_match = true;

            if(no_match)
            {
                static hipsparselt_internal_ostream& once
                    = hipsparselt_cerr
                      << "\nrocsparselt warning: No paths matched " << dir
                      << ". Make sure that ROCSPARSELT_TENSILE_LIBPATH is set correctly."
                      << std::endl;
            }

            hipDeviceProp_t prop;

            THROW_IF_HIP_ERROR(hipGetDeviceProperties(&prop, deviceId));

            m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
        }
    };

    // Return the library and adapter for the current HIP device
    auto& get_adapter(std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr, int device = -1)
    {
        try
        {
            // KernelLauncher is initialized on the first call
            static KernelLauncher host;

            if(device == -1)
                THROW_IF_HIP_ERROR(hipGetDevice(&device));

            // Adapter entry for the current HIP device ID
            auto& a       = host.get_adapters().at(device);
            auto* adapter = a.adapter.load(std::memory_order_acquire);

            // Once set, a.adapter contains the adapter for the current HIP device ID
            if(!adapter)
            {
                // Lock so that only one thread performs initialization of the adapter
                std::lock_guard<std::mutex> lock(a.mutex);

                adapter = a.adapter.load(std::memory_order_relaxed);
                if(!adapter)
                {
                    // Allocate a new adapter using the current HIP device
                    adapter = new SolutionAdapter();

                    // Initialize the adapter and possibly the library
                    host.initialize(*adapter, device);

                    // Atomically change the adapter stored for this device ID
                    a.adapter.store(adapter, std::memory_order_release);
                }
            }

            if(deviceProp)
                *deviceProp = host.get_device_property();

            return *adapter;
        }
        catch(const std::exception& e)
        {
            hipsparselt_cerr << "\nrocsparselt error: Could not initialize Kernel Launcher host:\n"
                             << e.what() << std::endl;
            hipsparselt_abort();
        }
        catch(...)
        {
            hipsparselt_cerr
                << "\nrocsparselt error: Could not initialize Kernel Launcher host:\nUnknown "
                   "exception thrown"
                << std::endl;
            hipsparselt_abort();
        }
    }

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const hipsparselt_internal_ostream& msg)
    {
        if(rocsparselt_suppress_kl_error_messages())
            return;
        static constexpr char varname[] = "ROCSPARSELT_VERBOSE_KL_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = hipsparselt_cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            hipsparselt_cerr << msg << std::endl;
    }

} // namespace

/******************************************************************************
 * runContractionProblem used to run a contraction problem described          *
 * by RocsparseltContractionProblem                                           *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status runContractionProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                         int*                                             config_id,
                                         const int config_max_id,
                                         const int search_iterations)
{
    rocsparselt_status status  = rocsparselt_status_internal_error;
    size_t             max_cid = 0;
    try
    {
        std::shared_ptr<hipDeviceProp_t> deviceProp;

        auto&       adapter = get_adapter(&deviceProp, prob.handle->device);
        std::string str     = generate_kernel_category_str<Ti, To, Tc>(prob.trans_a, prob.trans_b);
        max_cid             = adapter.getKernelCounts(str);
        KernelParams* solution = adapter.getKernelParams(str);

        if(config_max_id != max_cid)
        {
            hipsparselt_cerr << "config_max_id (" << config_max_id << ") is out of range ("
                             << max_cid << ") used this value to instead." << std::endl;
        }

        if(!max_cid)
        {
            hipsparselt_internal_ostream msg;
            print_once(msg << "\nrocsparselt error: No solution found for " << prob);
            status = rocsparselt_status_not_implemented;
        }
        else
        {
            if(!search_iterations)
            {
                RETURN_IF_HIP_ERROR(adapter.launchKernel(
                    prob.handle,
                    ConstructKernelInvoke<Ti, To, Tc>(prob, solution[*config_id]),
                    prob.streams[0],
                    nullptr,
                    nullptr));
            }
            else
            {
                float      min_ms = std::numeric_limits<float>::max();
                hipEvent_t startEvent, stopEvent;
                float      ms;
                RETURN_IF_HIP_ERROR(hipEventCreate(&startEvent));
                RETURN_IF_HIP_ERROR(hipEventCreate(&stopEvent));
                for(int id = 0; id < max_cid; id++)
                {
                    auto ki = ConstructKernelInvoke<Ti, To, Tc>(prob, solution[id]);
                    //warm up
                    RETURN_IF_HIP_ERROR(
                        adapter.launchKernel(prob.handle, ki, prob.streams[0], nullptr, nullptr));

                    RETURN_IF_HIP_ERROR(adapter.launchKernel(prob.handle,
                                                             ki,
                                                             prob.streams[0],
                                                             startEvent,
                                                             stopEvent,
                                                             search_iterations));
                    RETURN_IF_HIP_ERROR(hipEventSynchronize(stopEvent));
                    RETURN_IF_HIP_ERROR(hipEventElapsedTime(&ms, startEvent, stopEvent));
                    if(ms < min_ms)
                    {
                        *config_id = id;
                        min_ms = ms;
                    }

                }
                RETURN_IF_HIP_ERROR(hipEventDestroy(startEvent));
                RETURN_IF_HIP_ERROR(hipEventDestroy(stopEvent));
            }
            status = rocsparselt_status_success;
        }
    }
    catch(const std::exception& e)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nrocsparselt error: " << (max_cid ? "" : "No ")
                       << "Solution found, but exception thrown for " << prob << e.what());
    }
    catch(...)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nrocsparselt error: " << (max_cid ? "" : "No ")
                       << "Solution found, but unknown exception thrown for " << prob);
    }

    return status;
}

/******************************************************************************
 * initSolutions used to initialize specific type's solutions at the early stage.               *
 * ****************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status initSolutions(const _rocsparselt_handle* handle,
                                 rocsparselt_operation      opA,
                                 rocsparselt_operation      opB,
                                 int*                       kernel_counts)
{
    std::shared_ptr<hipDeviceProp_t> deviceProp;
    auto&                            adapter = get_adapter(&deviceProp, handle->device);
    std::string                      str     = generate_kernel_category_str<Ti, To, Tc>(opA, opB);

    *kernel_counts = adapter.getKernelCounts(str);
    if(*kernel_counts <= 0)
        return rocsparselt_status_not_implemented;

    KernelParams* solution = adapter.getKernelParams(str);
    for(int i = 0; i < *kernel_counts; i++)
        PRINT_IF_HIP_ERROR(handle, adapter.loadCodeObject(handle, solution[i].SolutionNameMin));
    return rocsparselt_status_success;
}

/***************************************************************
 * ! \brief  Initialize rocsparselt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocsparselt_initialize()
{
    get_adapter();
}

/*******************************************************************************************
 * Whether Kernel Launcher has been initialized for at least one device (used for testing) *
 *******************************************************************************************/
std::atomic_bool& rocsparselt_internal_kl_is_initialized()
{
    static std::atomic_bool init;
    return init;
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem / initSolutions which are    *
 * needed to satisfy rocsparselt dependencies.                                *
 ******************************************************************************/
#define GENERATE_DEFINITIONS(Ti, To, Tc, Ca)                                           \
    template <>                                                                        \
    std::string generate_kernel_category_str<Ti, To, Tc>(rocsparselt_operation opA,    \
                                                         rocsparselt_operation opB)    \
    {                                                                                  \
        std::string str = Ca;                                                          \
        str += "_";                                                                    \
        str += (opA == rocsparselt_operation_none ? "N" : "T");                        \
        str += "_";                                                                    \
        str += (opB == rocsparselt_operation_none ? "N" : "T");                        \
        return str;                                                                    \
    }                                                                                  \
    template rocsparselt_status runContractionProblem<Ti, To, Tc>(                     \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int*, const int, const int); \
    template rocsparselt_status initSolutions<Ti, To, Tc>(                             \
        const _rocsparselt_handle*, rocsparselt_operation, rocsparselt_operation, int*);

GENERATE_DEFINITIONS(__half, __half, float, "4_4_0")
GENERATE_DEFINITIONS(hip_bfloat16, hip_bfloat16, float, "7_7_0")
GENERATE_DEFINITIONS(int8_t, int8_t, float, "8_8_0")

.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, cuTensor, ROCm, library, API, tool

.. _Plan Cache:

=============================
Plan cache
=============================

Creating a plan for specific operations can be time-consuming, as it requires invoking performance models to determine the optimal solution. Therefore, it is advantageous to store the plan in memory cache for reuse in subsequent executions. The hipTensor library includes a software-managed plan cache designed to boost performance by reusing optimized execution plans across repeated tensor operations. This section explores its features and demonstrates how to use and customize the cache effectively.

--------------------------------
Key features
--------------------------------

- **Reduced Launch Overhead:** Reuses plans to minimize time spent on kernel selection and setup.
- **Autotuning (Incremental Autotuning):** Automatically benchmarks multiple kernel candidates to select the best-performing one.
- **Thread-Safe Design:** The cache is safe for concurrent use and is shared across all threads operating under a single `hiptensorHandle_t  <../api-reference/api-reference.html#hiptensorHandle>`_.
- **Persistence Support:** The cache state can be saved to disk and reloaded in future runs, avoiding repeated tuning.

At its core, the plan cache maps a specific problem configuration (represented by `hiptensorOperationDescriptor_t <../api-reference/api-reference.html#hiptensorOperationDescriptor>`_) to an optimized execution plan (`hiptensorPlan_t <../api-reference/api-reference.html#hiptensorPlan>`_).

------------------------------------
Default environment setting
------------------------------------

The plan cache is enabled by default. It can be disabled by setting the ``HIPTENSOR_DISABLE_PLAN_CACHE`` environment variable.
To disable the plan cache, set ``HIPTENSOR_DISABLE_PLAN_CACHE`` to ``ON``.
::

   export HIPTENSOR_DISABLE_PLAN_CACHE = ON

To enable the plan cache, set ``HIPTENSOR_DISABLE_PLAN_CACHE`` to ``OFF``.
::

   export HIPTENSOR_DISABLE_PLAN_CACHE = OFF

------------------------------------
Incremental autotuning
------------------------------------
Incremental autotuning is a feature that allows hipTensor to intelligently search for the most efficient implementation of a tensor operation without introducing measurable overhead.
When ``HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL`` is enabled, repeated executions of the same operation, even with different memory addresses, are tried with multiple backend kernels. Each candidate is measured automatically, and the fastest one is stored in the plan cache for subsequent use.
You can control the number of candidates explored using ``HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT``. For best results, it's recommended to warm up the GPU before autotuning to reduce performance variability.

*****************************
Advantages of autotuning
*****************************

- **Minimal Overhead:** No explicit timing loops or extra synchronization steps.
- **Realistic Benchmarking:** Candidate timings reflect production cache states, not cold-start measurements.
- **Smart Candidate Ordering:** Evaluations follow a performance model that prioritizes likely best options first.

------------------------------------
Save and load from disk
------------------------------------

When paired with cache serialization APIs like `hiptensorHandleWritePlanCacheToFile() <../api-reference/api-reference.html#hiptensorhandlewriteplancachetofile>`_ and `hiptensorHandleReadPlanCacheFromFile() <../api-reference/api-reference.html#hiptensorhandlereadplancachefromfile>`_, plan cache lines can be saved to disk and reloaded later to avoid the repeated benchmarking in future application runs.

------------------------------------
Plan cache example
------------------------------------

This example demonstrates how to set up the plan cache beyond default settings, including how to resize the cache, configure plan preferences and so on for individual contractions.

*****************************
1. Setting the cache size
*****************************
The default maximum cache line number is 128 and the number of entries is user-configurable:
::

    uint32_t numEntries = 256;
    CHECK_HIPTENSOR_ERROR(hiptensorHandleResizePlanCache(handle, numEntries));

Ideally, the cache should be large enough to store all unique contractions your application performs. If it exceeds capacity, hipTensor evicts entries using a Least Recently Used (LRU) policy.

**********************************************************
2. Disabling caching for specific contractions
**********************************************************
You can selectively disable caching for certain operations via the plan preference API:
::

    const hiptensorCacheMode_t cacheMode = HIPTENSOR_CACHE_MODE_NONE;
    HANDLE_ERROR(hiptensorPlanPreferenceSetAttribute(&handle,
                                                     &find,
                                                     HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                                                     &cacheMode,
                                                     sizeof(hiptensorCacheMode_t)));

Plan cache lookups occur during plan creation. Disabling the cache for frequent, identical contractions might lead to performance penalties.

***********************************
3. Enabling incremental autotuning
***********************************
To enable autotuning, use API `hiptensorPlanPreferenceSetAttribute <../api-reference/api-reference.html#hiptensorplanpreferencesetattribute>`_ as follows:
::

    const hiptensorAutotuneMode_t autotuneMode = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
    CHECK_HIPTENSOR_ERROR(
        hiptensorPlanPreferenceSetAttribute(handle,
                                            planPref,
                                            HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
                                            &autotuneMode,
                                            sizeof(hiptensorAutotuneMode_t)));
    const uint32_t incCount = 4;
    CHECK_HIPTENSOR_ERROR(
        hiptensorPlanPreferenceSetAttribute(handle,
                                            planPref,
                                            HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
                                            &incCount,
                                            sizeof(uint32_t)));

In the code above:

- The first attribute enables incremental tuning.
- The second sets the number of kernel variants to evaluate before caching the best result.

Using a value of around four provides good coverage with minimal initial overhead. Higher values can improve performance by checking more kernel candidates in the ranked solution system. If the tuned plan is used frequently, the initial overhead can be amortized over time.

************************************************
4. Using tags to distinguish similar operations
************************************************
In performance-critical scenarios, two contractions with identical descriptors can still perform differently due to differences in hardware cache states. Assigning unique tags ensures each operation is tuned independently.
This is particularly useful when:

- One operand has just been written or accessed by a previous call.
- The contraction is bandwidth-bound and cache-sensitive.

************************************************
5. Saving and loading the plan cache
************************************************
After tuning, you can store the optimized plan cache lines to disk and reload them during future runs:
::

    /************************************************
   * Load Plan Cache from disk
   ************************************************/
    const char        planCacheFileName[] = "./plan_cache.bin";
    uint32_t          numCachelines       = 0;
    hiptensorStatus_t status
        = hiptensorHandleReadPlanCacheFromFile(handle, planCacheFileName, &numCachelines);
    if(status == HIPTENSOR_STATUS_IO_ERROR)
    {
        std::cout << "File " << planCacheFileName << " doesn't seem to exist." << std::endl;
    }
    else if(status != HIPTENSOR_STATUS_SUCCESS)
    {
        std::cout << "hiptensorHandleReadPlanCacheFromFile reports error: "
                  << hiptensorGetErrorString(status) << std::endl;
    }
    else
    {
        std::cout << "hiptensorHandleReadPlanCacheFromFile read " << numCachelines
                  << " cachelines from file." << std::endl;
    }

::

    /**************************
   * Write Plan Cache to disk
   **************************/
    status = hiptensorHandleWritePlanCacheToFile(handle, planCacheFileName);
    if(status == HIPTENSOR_STATUS_IO_ERROR)
    {
        std::cout << "Plan Cache couldn't be written to " << planCacheFileName << std::endl;
    }
    else if(status != HIPTENSOR_STATUS_SUCCESS)
    {
        std::cout << "hiptensorHandleWritePlanCacheToFile reports error: "
                  << hiptensorGetErrorString(status) << std::endl;
    }
    else
    {
        std::cout << "Plan Cache successfully written to " << planCacheFileName << std::endl;
    }

This is especially valuable for:

- Applications that require consistent startup performance.
- Systems where tuning incurs a significant cost. For example, large candidate counts.

Complete code:
::

    #include <algorithm>
    #include <fstream>
    #include <hiptensor/hiptensor.h>
    #include <hiptensor/hiptensor_types.h>
    #include <hiptensor/internal/hiptensor_utility.hpp>
    #include <iterator>
    #include <numeric>
    #include <unordered_map>

    #include "common.hpp"

    int main(int argc, char* argv[])
    {
        /***************************************
       * Check device support                 *
       **************************************/
        if(!isF32Supported())
        {
            std::cout << "unsupported host device" << std::endl;
            exit(EXIT_FAILURE);
        }

        typedef float ADataType;
        typedef float BDataType;
        typedef float CDataType;
        typedef float floatTypeCompute;

        constexpr hiptensorDataType_t          typeA       = HIPTENSOR_R_32F;
        constexpr hiptensorDataType_t          typeB       = HIPTENSOR_R_32F;
        constexpr hiptensorDataType_t          typeC       = HIPTENSOR_R_32F;
        constexpr hiptensorComputeDescriptor_t typeCompute = HIPTENSOR_COMPUTE_DESC_32F;

        floatTypeCompute alpha{2.3f};
        floatTypeCompute beta{1.1f};

        /**********************
       * Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta *
       *C_{m,n,u,v}
       **********************/

        std::vector<int> modeC{'m', 'n', 'u', 'v'};
        std::vector<int> modeA{'m', 'n', 'h', 'k'};
        std::vector<int> modeB{'u', 'v', 'h', 'k'};

        int nmodeA = modeA.size();
        int nmodeB = modeB.size();
        int nmodeC = modeC.size();

        std::unordered_map<int, int64_t> extent;

        extent['m'] = 96;
        extent['n'] = 96;
        extent['u'] = 96;
        extent['v'] = 64;
        extent['h'] = 64;
        extent['k'] = 64;

        std::vector<int64_t> c_ms_ns_lengths;
        for(auto mode : modeC)
        {
            c_ms_ns_lengths.push_back(extent[mode]);
        }

        std::vector<int64_t> a_ms_ks_lengths;
        for(auto mode : modeA)
        {
            a_ms_ks_lengths.push_back(extent[mode]);
        }

        std::vector<int64_t> b_ns_ks_lengths;
        for(auto mode : modeB)
        {
            b_ns_ks_lengths.push_back(extent[mode]);
        }

        /**********************
       * Allocating data
       **********************/
        std::cout << "Initializing host data..." << std::endl;

        size_t elementsA = std::accumulate(
            a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
        size_t elementsB = std::accumulate(
            b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
        size_t elementsC = std::accumulate(
            c_ms_ns_lengths.begin(), c_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

        size_t sizeA = sizeof(ADataType) * elementsA;
        size_t sizeB = sizeof(BDataType) * elementsB;
        size_t sizeC = sizeof(CDataType) * elementsC;

        ADataType* A = nullptr;
        BDataType* B = nullptr;
        CDataType* C = nullptr;
        CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
        CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));
        CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC));

        void *A_d, *B_d, *C_d;

        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&C_d), sizeC));

        /*******************
       * Initialize data
       *******************/
        int initMethod = 1; // TODO read value from commandline
        for(int64_t i = 0; i < elementsA; i++)
        {
            if(initMethod == 0)
            {
                A[i] = ADataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
            }
            else
            {
                A[i] = (ADataType)(float(i) / 100);
            }
        }

        for(int64_t i = 0; i < elementsB; i++)
        {
            if(initMethod == 0)
            {
                B[i] = BDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
            }
            else
            {
                B[i] = (BDataType)(float(i) / 100);
            }
        }

        for(int64_t i = 0; i < elementsC; i++)
        {
            if(initMethod == 0)
            {
                C[i] = CDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
            }
            else
            {
                C[i] = (BDataType)(float(i) / 100);
            }
        }

        /********************************************
       * Transfer the Host Tensor to Device Memory *
       ********************************************/
        std::cout << "Initializing device data..." << std::endl;

        CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

        /************************************************
       * Retrieve the memory alignment for each tensor
       ************************************************/
        uint32_t          alignmentRequirement = 1;
        hiptensorHandle_t handle;
        CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

        CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

        /************************************************
       * load Plan Cache from disk
       ************************************************/
        const char        planCacheFileName[] = "./plan_cache.bin";
        uint32_t          numCachelines       = 0;
        hiptensorStatus_t status
            = hiptensorHandleReadPlanCacheFromFile(handle, planCacheFileName, &numCachelines);
        if(status == HIPTENSOR_STATUS_IO_ERROR)
        {
            std::cout << "File " << planCacheFileName << " doesn't seem to exist." << std::endl;
        }
        else if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            std::cout << "hiptensorHandleReadPlanCacheFromFile reports error: "
                      << hiptensorGetErrorString(status) << std::endl;
        }
        else
        {
            std::cout << "hiptensorHandleReadPlanCacheFromFile read " << numCachelines
                      << " cachelines from file." << std::endl;
        }

        /**********************
        * Optional: Resize the maximum number of cache lines
        **********************/
        uint32_t numEntries = 128;
        CHECK_HIPTENSOR_ERROR(hiptensorHandleResizePlanCache(handle, numEntries));

        /********************************************
       * Initialize tensors with the input lengths *
       ********************************************/
        hiptensorTensorDescriptor_t a_ms_ks = nullptr;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &a_ms_ks,
                                                              nmodeA,
                                                              a_ms_ks_lengths.data(),
                                                              NULL, /*stride*/
                                                              typeA,
                                                              alignmentRequirement));

        hiptensorTensorDescriptor_t b_ns_ks = nullptr;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &b_ns_ks,
                                                              nmodeB,
                                                              b_ns_ks_lengths.data(),
                                                              NULL, /*stride*/
                                                              typeB,
                                                              alignmentRequirement));

        hiptensorTensorDescriptor_t c_ms_ns = nullptr;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &c_ms_ns,
                                                              nmodeC,
                                                              c_ms_ns_lengths.data(),
                                                              NULL, /*stride*/
                                                              typeC,
                                                              alignmentRequirement));

        /*******************************
       * Create Contraction Descriptor
       *******************************/

        hiptensorOperationDescriptor_t desc;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(handle,
                                                         &desc,
                                                         a_ms_ks,
                                                         modeA.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         b_ns_ks,
                                                         modeB.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         c_ms_ns,
                                                         modeC.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         c_ms_ns,
                                                         modeC.data(),
                                                         typeCompute));

        /**************************
       * Set the algorithm to use
       ***************************/
        hiptensorPlanPreference_t planPref;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
            handle, &planPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

        const hiptensorCacheMode_t cacheMode = HIPTENSOR_CACHE_MODE_PEDANTIC;
        CHECK_HIPTENSOR_ERROR(hiptensorPlanPreferenceSetAttribute(handle,
                                                                  planPref,
                                                                  HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                                                                  &cacheMode,
                                                                  sizeof(hiptensorCacheMode_t)));

        const hiptensorAutotuneMode_t autotuneMode = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
        CHECK_HIPTENSOR_ERROR(
            hiptensorPlanPreferenceSetAttribute(handle,
                                                planPref,
                                                HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
                                                &autotuneMode,
                                                sizeof(hiptensorAutotuneMode_t)));

        const uint32_t incCount = 4;
        CHECK_HIPTENSOR_ERROR(
            hiptensorPlanPreferenceSetAttribute(handle,
                                                planPref,
                                                HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
                                                &incCount,
                                                sizeof(uint32_t)));

        /**********************
       * Query workspace
       **********************/

        uint64_t worksize = 0;
        CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
            handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

        /**********************
        * Optional: Set a different tag
        **********************/
        uint32_t tag = 1u;
        CHECK_HIPTENSOR_ERROR(hiptensorOperationDescriptorSetAttribute(
            handle, desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(uint32_t)));

        /**************************
       * Create Contraction Plan
       **************************/
        std::cout << "Initializing contraction plan..." << std::endl;

        hiptensorPlan_t plan;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

        // TODO query actually used workspace
        void* workspace = nullptr;

        if(worksize > 0)
        {
            CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
        }

        std::cout << "Launching contraction kernel..." << std::endl;

        for(int i = 0; i < incCount + 1; i++) // last iteration will hit the cache
        {
            CHECK_HIPTENSOR_ERROR(hiptensorContract(
                handle, plan, &alpha, A_d, B_d, &beta, C_d, C_d, workspace, worksize, 0 /* stream */));
        }

        /**************************
       * Write Plan Cache to disk
       **************************/
        status = hiptensorHandleWritePlanCacheToFile(handle, planCacheFileName);
        if(status == HIPTENSOR_STATUS_IO_ERROR)
        {
            std::cout << "Plan Cache couldn't be written to " << planCacheFileName << std::endl;
        }
        else if(status != HIPTENSOR_STATUS_SUCCESS)
        {
            std::cout << "hiptensorHandleWritePlanCacheToFile reports error: "
                      << hiptensorGetErrorString(status) << std::endl;
        }
        else
        {
            std::cout << "Plan Cache successfully written to " << planCacheFileName << std::endl;
        }

        CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
        if(a_ms_ks)
        {
            hiptensorDestroyTensorDescriptor(a_ms_ks);
            a_ms_ks = nullptr;
        }
        if(b_ns_ks)
        {
            hiptensorDestroyTensorDescriptor(b_ns_ks);
            b_ns_ks = nullptr;
        }
        if(c_ms_ns)
        {
            hiptensorDestroyTensorDescriptor(c_ms_ns);
            c_ms_ns = nullptr;
        }

        HIPTENSOR_FREE_HOST(A);
        HIPTENSOR_FREE_HOST(B);
        HIPTENSOR_FREE_HOST(C);

        HIPTENSOR_FREE_DEVICE(A_d);
        HIPTENSOR_FREE_DEVICE(B_d);
        HIPTENSOR_FREE_DEVICE(C_d);
        HIPTENSOR_FREE_DEVICE(workspace);

        std::cout << "Finished!" << std::endl;

        return 0;
    }

------------------------------------
Summary
------------------------------------
The plan cache and incremental autotuning in hipTensor provide a robust framework for performance tuning and optimization. By leveraging these tools, you can significantly reduce runtime overhead and improve performance in compute-intensive tensor operations.

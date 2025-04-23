/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

#include "backend_hip.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "hip_utils.hpp"

#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_dense.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_vector.hpp"

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#include <complex>

namespace rocalution
{
    bool rocalution_init_hip(void)
    {
        log_debug(0, "rocalution_init_hip()", "* begin");

        assert(_get_backend_descriptor()->ROC_blas_handle == NULL);
        assert(_get_backend_descriptor()->ROC_sparse_handle == NULL);
        //  assert(_get_backend_descriptor()->HIP_dev == -1);

        // create a handle
        _get_backend_descriptor()->ROC_blas_handle   = new rocblas_handle;
        _get_backend_descriptor()->ROC_sparse_handle = new rocsparse_handle;

        // get last error (if any)
        hipGetLastError();

        // HIP streams
        assert(_get_backend_descriptor()->HIP_stream_default == NULL);
        assert(_get_backend_descriptor()->HIP_stream_interior == NULL);
        assert(_get_backend_descriptor()->HIP_stream_ghost == NULL);
        assert(_get_backend_descriptor()->HIP_stream_current == NULL);

        // create streams
        _get_backend_descriptor()->HIP_stream_default = new hipStream_t;
#ifdef SUPPORT_MULTINODE
        _get_backend_descriptor()->HIP_stream_interior = new hipStream_t;
        _get_backend_descriptor()->HIP_stream_ghost    = new hipStream_t;
#endif

        // default stream is null stream
        *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_default)) = NULL;

        _get_backend_descriptor()->HIP_stream_current
            = _get_backend_descriptor()->HIP_stream_default;

        hipError_t hip_status_t;
        int        num_dev;
        hipGetDeviceCount(&num_dev);
        hip_status_t = hipGetLastError();

        // if querying for device count fails, fall back to host backend
        if(hip_status_t != hipSuccess)
        {
            LOG_INFO("Querying for HIP devices failed - falling back to host backend");
            return false;
        }

        LOG_INFO("Number of HIP devices in the system: " << num_dev);

        if(num_dev < 1)
        {
            LOG_INFO("No HIP device found");
        }
        else
        {
            if(_get_backend_descriptor()->HIP_dev != -1)
            {
                num_dev = 1;
            }

            for(int idev = 0; idev < num_dev; idev++)
            {
                int dev = idev;

                if(_get_backend_descriptor()->HIP_dev != -1)
                {
                    dev = _get_backend_descriptor()->HIP_dev;
                }

                hipSetDevice(dev);
                hip_status_t = hipGetLastError();

                if(hip_status_t == hipErrorContextAlreadyInUse)
                {
                    LOG_INFO("HIP context of device " << dev << " is already in use");
                    return false;
                }

                if(hip_status_t == hipErrorInvalidDevice)
                {
                    LOG_INFO("HIP device " << dev << " is invalid");
                    return false;
                }

                if(hip_status_t == hipSuccess)
                {
                    if((rocblas_create_handle(static_cast<rocblas_handle*>(
                            _get_backend_descriptor()->ROC_blas_handle))
                        == rocblas_status_success)
                       && (rocsparse_create_handle(static_cast<rocsparse_handle*>(
                               _get_backend_descriptor()->ROC_sparse_handle))
                           == rocsparse_status_success)
#ifdef SUPPORT_MULTINODE
                       && (hipStreamCreate(static_cast<hipStream_t*>(
                               _get_backend_descriptor()->HIP_stream_interior))
                           == hipSuccess)
                       && (hipStreamCreate(static_cast<hipStream_t*>(
                               _get_backend_descriptor()->HIP_stream_ghost))
                           == hipSuccess)
#endif
                    )
                    {
                        _get_backend_descriptor()->HIP_dev = dev;
                        break;
                    }
                    else
                    {
                        LOG_INFO("HIP device "
                                 << dev
                                 << " cannot create rocBLAS/rocSPARSE context and HIP streams");
                    }
                }
            }
        }

        if(_get_backend_descriptor()->HIP_dev == -1)
        {
            LOG_INFO("HIP and rocBLAS/rocSPARSE have NOT been initialized!");
            return false;
        }

        struct hipDeviceProp_t dev_prop;
        hipGetDeviceProperties(&dev_prop, _get_backend_descriptor()->HIP_dev);

        if(dev_prop.major < 3)
        {
            LOG_INFO("HIP device " << _get_backend_descriptor()->HIP_dev
                                   << " has low compute capability (min 3.0 is needed)");
            return false;
        }

        // Get some properties from the device
        _get_backend_descriptor()->HIP_warp             = dev_prop.warpSize;
        _get_backend_descriptor()->HIP_num_procs        = dev_prop.multiProcessorCount;
        _get_backend_descriptor()->HIP_threads_per_proc = dev_prop.maxThreadsPerMultiProcessor;
        _get_backend_descriptor()->HIP_max_threads
            = dev_prop.regsPerBlock > 0 ? dev_prop.regsPerBlock : 65536;

        log_debug(0, "rocalution_init_hip()", "* end");

        return true;
    }

    void rocalution_stop_hip(void)
    {
        log_debug(0, "rocalution_stop_hip()", "* begin");

        if(_get_backend_descriptor()->accelerator)
        {
            if(rocblas_destroy_handle(
                   *(static_cast<rocblas_handle*>(_get_backend_descriptor()->ROC_blas_handle)))
               != rocblas_status_success)
            {
                LOG_INFO("Error in rocblas_destroy_handle");
            }

            if(rocsparse_destroy_handle(
                   *(static_cast<rocsparse_handle*>(_get_backend_descriptor()->ROC_sparse_handle)))
               != rocsparse_status_success)
            {
                LOG_INFO("Error in rocsparse_destroy_handle");
            }

#ifdef SUPPORT_MULTINODE
            if(hipStreamDestroy(
                   *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_interior)))
               != hipSuccess)
            {
                LOG_INFO("Error in hipStreamDestroy");
            }

            if(hipStreamDestroy(
                   *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_ghost)))
               != hipSuccess)
            {
                LOG_INFO("Error in hipStreamDestroy");
            }
#endif
        }

        delete(static_cast<rocblas_handle*>(_get_backend_descriptor()->ROC_blas_handle));
        delete(static_cast<rocsparse_handle*>(_get_backend_descriptor()->ROC_sparse_handle));

        _get_backend_descriptor()->ROC_blas_handle   = NULL;
        _get_backend_descriptor()->ROC_sparse_handle = NULL;

        delete(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_default));
#ifdef SUPPORT_MULTINODE
        delete(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_interior));
        delete(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_ghost));
#endif

        _get_backend_descriptor()->HIP_stream_default  = NULL;
        _get_backend_descriptor()->HIP_stream_interior = NULL;
        _get_backend_descriptor()->HIP_stream_ghost    = NULL;
        _get_backend_descriptor()->HIP_stream_current  = NULL;

        _get_backend_descriptor()->HIP_dev = -1;

        log_debug(0, "rocalution_stop_hip()", "* end");
    }

    void rocalution_info_hip(const struct Rocalution_Backend_Descriptor& backend_descriptor)
    {
        // Print rocblas version
        char rocblas_ver[64];
        rocblas_get_version_string(rocblas_ver, sizeof(rocblas_ver));

        LOG_INFO("rocBLAS ver " << rocblas_ver);

        // Print rocsparse versions
        int rocsparse_ver;
        rocsparse_get_version(ROCSPARSE_HANDLE(_get_backend_descriptor()->ROC_sparse_handle),
                              &rocsparse_ver);

        char rocsparse_rev[64];
        rocsparse_get_git_rev(ROCSPARSE_HANDLE(_get_backend_descriptor()->ROC_sparse_handle),
                              rocsparse_rev);

        LOG_INFO("rocSPARSE ver " << rocsparse_ver / 100000 << "." << rocsparse_ver / 100 % 1000
                                  << "." << rocsparse_ver % 100 << "-" << rocsparse_rev);

        int num_dev;

        hipGetDeviceCount(&num_dev);
        hipGetLastError();
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        if(_get_backend_descriptor()->HIP_dev >= 0)
        {
            // clang-format off
            LOG_INFO("------------------------------------------------");
            LOG_INFO("Selected HIP device: " << backend_descriptor.HIP_dev);

            struct hipDeviceProp_t dev_prop;
            hipGetDeviceProperties(&dev_prop, backend_descriptor.HIP_dev);

            // LOG_INFO("Device number: " << backend_descriptor.HIP_dev);
            LOG_INFO("Device name: " << dev_prop.name);                                        // char name[256];
            LOG_INFO("totalGlobalMem: " << (dev_prop.totalGlobalMem >> 20) << " MByte");       // size_t totalGlobalMem;
            /*
            LOG_INFO("sharedMemPerBlock: "           << dev_prop.sharedMemPerBlock);           // size_t sharedMemPerBlock;
            LOG_INFO("regsPerBlock: "                << dev_prop.regsPerBlock);                // int regsPerBlock;
            LOG_INFO("warpSize: "                    << dev_prop.warpSize);                    // int warpSize;
            LOG_INFO("memPitch: "                    << dev_prop.memPitch);                    // size_t memPitch;
            LOG_INFO("maxThreadsPerBlock: "          << dev_prop.maxThreadsPerBlock);          // int maxThreadsPerBlock;
            LOG_INFO("maxThreadsDim[0]: "            << dev_prop.maxThreadsDim[0]);            // int maxThreadsDim[0];
            LOG_INFO("maxThreadsDim[1]: "            << dev_prop.maxThreadsDim[1]);            // int maxThreadsDim[1];
            LOG_INFO("maxThreadsDim[2]: "            << dev_prop.maxThreadsDim[2]);            // int maxThreadsDim[2];
            LOG_INFO("maxGridSize[0]: "              << dev_prop.maxGridSize[0]);              // int maxGridSize[0];
            LOG_INFO("maxGridSize[1]: "              << dev_prop.maxGridSize[1]);              // int maxGridSize[1];
            LOG_INFO("maxGridSize[2]: "              << dev_prop.maxGridSize[2]);              // int maxGridSize[2];
            */
            LOG_INFO("clockRate: " << dev_prop.clockRate);                                     // int clockRate;
            /*
            LOG_INFO("totalConstMem: "               << dev_prop.totalConstMem);               // size_t totalConstMem;
            LOG_INFO("major: "                       << dev_prop.major);                       // int major;
            LOG_INFO("minor: "                       << dev_prop.minor);                       // int minor;
            */
            LOG_INFO("compute capability: " << dev_prop.major << "." << dev_prop.minor);
            /*
            LOG_INFO("textureAlignment: "            << dev_prop.textureAlignment);            // size_t textureAlignment;
            LOG_INFO("deviceOverlap: "               << dev_prop.deviceOverlap);               // int deviceOverlap;
            LOG_INFO("multiProcessorCount: "         << dev_prop.multiProcessorCount);         // int multiProcessorCount;
            LOG_INFO("kernelExecTimeoutEnabled: "    << dev_prop.kernelExecTimeoutEnabled);    // int kernelExecTimeoutEnabled;
            LOG_INFO("integrated: "                  << dev_prop.integrated);                  // int integrated;
            LOG_INFO("canMapHostMemory: "            << dev_prop.canMapHostMemory);            // int canMapHostMemory;
            LOG_INFO("computeMode: "                 << dev_prop.computeMode);                 // int computeMode;
            LOG_INFO("maxTexture1D: "                << dev_prop.maxTexture1D);                // int maxTexture1D;
            LOG_INFO("maxTexture2D[0]: "             << dev_prop.maxTexture2D[0]);             // int maxTexture2D[0];
            LOG_INFO("maxTexture2D[1]: "             << dev_prop.maxTexture2D[1]);             // int maxTexture2D[1];
            LOG_INFO("maxTexture3D[0]: "             << dev_prop.maxTexture3D[0]);             // int maxTexture3D[0];
            LOG_INFO("maxTexture3D[1]: "             << dev_prop.maxTexture3D[1]);             // int maxTexture3D[1];
            LOG_INFO("maxTexture3D[2]: "             << dev_prop.maxTexture3D[2]);             // int maxTexture3D[2];
            LOG_INFO("maxTexture1DLayered[0]: "      << dev_prop.maxTexture1DLayered[0]);      // int maxTexture1DLayered[0];
            LOG_INFO("maxTexture1DLayered[1]: "      << dev_prop.maxTexture1DLayered[1]);      // int maxTexture1DLayered[1];
            LOG_INFO("maxTexture2DLayered[0]: "      << dev_prop.maxTexture2DLayered[0]);      // int maxTexture2DLayered[0];
            LOG_INFO("maxTexture2DLayered[1]: "      << dev_prop.maxTexture2DLayered[1]);      // int maxTexture2DLayered[1];
            LOG_INFO("maxTexture2DLayered[2]: "      << dev_prop.maxTexture2DLayered[2]);      // int maxTexture2DLayered[2];
            LOG_INFO("surfaceAlignment: "            << dev_prop.surfaceAlignment);            // size_t surfaceAlignment;
            LOG_INFO("concurrentKernels: "           << dev_prop.concurrentKernels);           // int concurrentKernels;
            LOG_INFO("ECCEnabled: "                  << dev_prop.ECCEnabled);                  // int ECCEnabled;
            LOG_INFO("pciBusID: "                    << dev_prop.pciBusID);                    // int pciBusID;
            LOG_INFO("pciDeviceID: "                 << dev_prop.pciDeviceID);                 // int pciDeviceID;
            LOG_INFO("pciDomainID: "                 << dev_prop.pciDomainID);                 // int pciDomainID;
            LOG_INFO("tccDriver: "                   << dev_prop.tccDriver);                   // int tccDriver;
            LOG_INFO("asyncEngineCount: "            << dev_prop.asyncEngineCount);            // int asyncEngineCount;
            LOG_INFO("unifiedAddressing: "           << dev_prop.unifiedAddressing);           // int unifiedAddressing;
            LOG_INFO("memoryClockRate: "             << dev_prop.memoryClockRate);             // int memoryClockRate;
            LOG_INFO("memoryBusWidth: "              << dev_prop.memoryBusWidth);              // int memoryBusWidth;
            LOG_INFO("l2CacheSize: "                 << dev_prop.l2CacheSize);                 // int l2CacheSize;
            LOG_INFO("maxThreadsPerMultiProcessor: " << dev_prop.maxThreadsPerMultiProcessor); // int maxThreadsPerMultiProcessor;
            */
            LOG_INFO("------------------------------------------------");
            // clang-format on
        }
        else
        {
            LOG_INFO("No HIP device is selected!");
        }
    }

    std::string rocalution_get_arch_hip(void)
    {
        struct hipDeviceProp_t dev_prop;
        hipGetDeviceProperties(&dev_prop, _get_backend_descriptor()->HIP_dev);
        return dev_prop.gcnArchName;
    }

    void rocalution_hip_compute_interior(void)
    {
#ifdef SUPPORT_MULTINODE
        _get_backend_descriptor()->HIP_stream_current
            = _get_backend_descriptor()->HIP_stream_interior;

        rocsparse_status status_sparse = rocsparse_set_stream(
            *(static_cast<rocsparse_handle*>(_get_backend_descriptor()->ROC_sparse_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCSPARSE_ERROR(status_sparse, __FILE__, __LINE__);

        rocblas_status status_blas = rocblas_set_stream(
            *(static_cast<rocblas_handle*>(_get_backend_descriptor()->ROC_blas_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCBLAS_ERROR(status_blas, __FILE__, __LINE__);
#endif
    }

    void rocalution_hip_compute_ghost(void)
    {
#ifdef SUPPORT_MULTINODE
        _get_backend_descriptor()->HIP_stream_current = _get_backend_descriptor()->HIP_stream_ghost;

        rocsparse_status status_sparse = rocsparse_set_stream(
            *(static_cast<rocsparse_handle*>(_get_backend_descriptor()->ROC_sparse_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCSPARSE_ERROR(status_sparse, __FILE__, __LINE__);

        rocblas_status status_blas = rocblas_set_stream(
            *(static_cast<rocblas_handle*>(_get_backend_descriptor()->ROC_blas_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCBLAS_ERROR(status_blas, __FILE__, __LINE__);
#endif
    }

    void rocalution_hip_compute_default(void)
    {
#ifdef SUPPORT_MULTINODE
        // Default stream is NULL
        _get_backend_descriptor()->HIP_stream_current
            = _get_backend_descriptor()->HIP_stream_default;

        rocsparse_status status_sparse = rocsparse_set_stream(
            *(static_cast<rocsparse_handle*>(_get_backend_descriptor()->ROC_sparse_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCSPARSE_ERROR(status_sparse, __FILE__, __LINE__);

        rocblas_status status_blas = rocblas_set_stream(
            *(static_cast<rocblas_handle*>(_get_backend_descriptor()->ROC_blas_handle)),
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_current)));
        CHECK_ROCBLAS_ERROR(status_blas, __FILE__, __LINE__);
#endif
    }

    template <typename ValueType>
    AcceleratorMatrix<ValueType>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim)
    {
        assert(backend_descriptor.backend == HIP);

        if(matrix_format != BCSR)
        {
            assert(blockdim == 1);
        }

        switch(matrix_format)
        {
        case CSR:
            return new HIPAcceleratorMatrixCSR<ValueType>(backend_descriptor);
        case COO:
            return new HIPAcceleratorMatrixCOO<ValueType>(backend_descriptor);
        case MCSR:
            return new HIPAcceleratorMatrixMCSR<ValueType>(backend_descriptor);
        case DIA:
            return new HIPAcceleratorMatrixDIA<ValueType>(backend_descriptor);
        case ELL:
            return new HIPAcceleratorMatrixELL<ValueType>(backend_descriptor);
        case DENSE:
            return new HIPAcceleratorMatrixDENSE<ValueType>(backend_descriptor);
        case HYB:
            return new HIPAcceleratorMatrixHYB<ValueType>(backend_descriptor);
        case BCSR:
            return new HIPAcceleratorMatrixBCSR<ValueType>(backend_descriptor, blockdim);
        // LCOV_EXCL_START
        default:
            LOG_INFO("This backed is not supported for Matrix types");
            FATAL_ERROR(__FILE__, __LINE__);
            return NULL;
            // LCOV_EXCL_STOP
        }
    }

    template <typename ValueType>
    AcceleratorVector<ValueType>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor)
    {
        assert(backend_descriptor.backend == HIP);
        return new HIPAcceleratorVector<ValueType>(backend_descriptor);
    }

    void rocalution_hip_sync(void)
    {
        hipDeviceSynchronize();
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    void rocalution_hip_sync_default(void)
    {
        hipStreamSynchronize(
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_default)));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    void rocalution_hip_sync_interior(void)
    {
#ifdef SUPPORT_MULTINODE
        hipStreamSynchronize(
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_interior)));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
#endif
    }

    void rocalution_hip_sync_ghost(void)
    {
#ifdef SUPPORT_MULTINODE
        hipStreamSynchronize(
            *(static_cast<hipStream_t*>(_get_backend_descriptor()->HIP_stream_ghost)));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
#endif
    }

    template AcceleratorVector<float>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<double>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
#ifdef SUPPORT_COMPLEX
    template AcceleratorVector<std::complex<float>>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<std::complex<double>>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
#endif
    template AcceleratorVector<bool>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<int>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<int64_t>* _rocalution_init_base_hip_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);

    template AcceleratorMatrix<float>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template AcceleratorMatrix<double>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#ifdef SUPPORT_COMPLEX
    template AcceleratorMatrix<std::complex<float>>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template AcceleratorMatrix<std::complex<double>>* _rocalution_init_base_hip_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#endif

} // namespace rocalution

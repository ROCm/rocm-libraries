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

#include "backend_manager.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "base_matrix.hpp"
#include "base_rocalution.hpp"
#include "base_vector.hpp"
#include "host/host_affinity.hpp"
#include "host/host_matrix_bcsr.hpp"
#include "host/host_matrix_coo.hpp"
#include "host/host_matrix_csr.hpp"
#include "host/host_matrix_dense.hpp"
#include "host/host_matrix_dia.hpp"
#include "host/host_matrix_ell.hpp"
#include "host/host_matrix_hyb.hpp"
#include "host/host_matrix_mcsr.hpp"
#include "host/host_vector.hpp"
#include "rocalution/version.hpp"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SUPPORT_HIP
#include "hip/backend_hip.hpp"
#endif

#ifdef SUPPORT_MULTINODE
#include "../utils/log_mpi.hpp"
#include <mpi.h>
#endif

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

namespace rocalution
{

    // Global backend descriptor and default values
    Rocalution_Backend_Descriptor _Backend_Descriptor = {
        false, // Init
#ifdef SUPPORT_HIP
        HIP, // default backend
#else
        None,
#endif
        false, // use accelerator
        false, // disable accelerator
        1, // OpenMP threads
        -1, // pre-init OpenMP threads
        0, // pre-init OpenMP threads
        true, // host affinity (active)
        10000, // threshold size
        // HIP section
        NULL, // *HIP_blas_handle
        NULL, // *HIP_sparse_handle
        NULL, // *HIP_stream_default
        NULL, // *HIP_stream_interior
        NULL, // *HIP_stream_ghost
        NULL, // *HIP_stream_current
        -1, // HIP_dev;
        32, // HIP_warp;
        256, // HIP_blocksize;
        65535, // Maximum threads in the block
        13, // HIP_num_procs
        2048, // HIP_threads_per_proc
        // MPI rank/id
        0,
        // LOG
        0,
        NULL // FILE, file log
    };

    /// Host name
    const std::string _rocalution_host_name[1] =
#ifdef _OPENMP
        {"CPU(OpenMP)"};
#else
        {"CPU"};
#endif

    /// Backend names
    const std::string _rocalution_backend_name[2] = {"None", "HIP"};

    int init_rocalution(int rank, int dev_per_node)
    {
        // please note your MPI communicator
        if(rank >= 0)
        {
            _get_backend_descriptor()->rank = rank;
        }
        else
        {
            int current_rank = 0;

#ifdef SUPPORT_MULTINODE
            int status_init;

            MPI_Initialized(&status_init);

            if(status_init == 1)
            {
                MPI_Comm comm   = MPI_COMM_WORLD;
                int      status = MPI_Comm_rank(comm, &current_rank);

                if(status != MPI_SUCCESS)
                {
                    current_rank = 0;
                }
            }
#endif

            _get_backend_descriptor()->rank = current_rank;
        }

        _rocalution_open_log_file();

        log_debug(0, "init_rocalution()", "* begin", rank, dev_per_node);

        if(_get_backend_descriptor()->init == true)
        {
            LOG_INFO("rocALUTION platform has been initialized - restarting");
            stop_rocalution();
        }

#ifdef SUPPORT_HIP
        _get_backend_descriptor()->backend = HIP;
#else
        _get_backend_descriptor()->backend = None;
#endif

#ifdef _OPENMP
        _get_backend_descriptor()->OpenMP_def_threads = omp_get_max_threads();
        _get_backend_descriptor()->OpenMP_threads     = omp_get_max_threads();
#if _OPENMP >= 201811
        _get_backend_descriptor()->OpenMP_def_nested = omp_get_max_active_levels();
#else
        _get_backend_descriptor()->OpenMP_def_nested = omp_get_nested();
#endif

        // the default in rocALUTION is 0
#if _OPENMP >= 201811
        omp_set_max_active_levels(0);
#else
        omp_set_nested(0);
#endif

        rocalution_set_omp_affinity(_get_backend_descriptor()->OpenMP_affinity);
#else
        _get_backend_descriptor()->OpenMP_threads = 1;
#endif

        if(_get_backend_descriptor()->disable_accelerator == false)
        {
#ifdef SUPPORT_HIP
            if(rank > -1 && dev_per_node > 0)
            {
                set_device_rocalution(rank % dev_per_node);
            }

            _get_backend_descriptor()->accelerator = rocalution_init_hip();

            if(_get_backend_descriptor()->accelerator == false)
            {
                LOG_INFO("Warning: the accelerator is disabled");
            }
#endif
        }
        else
        {
            LOG_INFO("Warning: the accelerator is disabled");
        }

        if(_rocalution_check_if_any_obj() == false)
        {
            LOG_INFO("Error: rocALUTION objects have been created before calling the "
                     "init_rocalution()!");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        _get_backend_descriptor()->init = true;

        log_debug(0, "init_rocalution()", "* end");

        return 0;
    }

    int stop_rocalution(void)
    {
        log_debug(0, "stop_rocalution()", "* begin");

        if(_get_backend_descriptor()->init == false)
        {
            return 0;
        }

        _rocalution_delete_all_obj();

#ifdef SUPPORT_HIP
        if(_get_backend_descriptor()->disable_accelerator == false)
        {
            rocalution_stop_hip();
        }
#endif

#ifdef _OPENMP
        assert(_get_backend_descriptor()->OpenMP_def_threads > 0);
        omp_set_num_threads(_get_backend_descriptor()->OpenMP_def_threads);

        assert((_get_backend_descriptor()->OpenMP_def_nested == 0)
               || (_get_backend_descriptor()->OpenMP_def_nested == 1));
#if _OPENMP >= 201811
        omp_set_max_active_levels(_get_backend_descriptor()->OpenMP_def_nested);
#else
        omp_set_nested(_get_backend_descriptor()->OpenMP_def_nested);
#endif
#endif

        _get_backend_descriptor()->init = false;

        log_debug(0, "stop_rocalution()", "* end");

        _rocalution_close_log_file();

        return 0;
    }

    void set_omp_threads_rocalution(int nthreads)
    {
        log_debug(0, "set_omp_threads_rocalution()", nthreads);

        assert(_get_backend_descriptor()->init == true);

#ifdef _OPENMP
        _get_backend_descriptor()->OpenMP_threads = nthreads;

        omp_set_num_threads(nthreads);

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__)

        rocalution_set_omp_affinity(_get_backend_descriptor()->OpenMP_affinity);

#endif // linux

#else // !omp
        LOG_INFO("No OpenMP support");
        _get_backend_descriptor()->OpenMP_threads = 1;
#endif // omp
    }

    void set_device_rocalution(int dev)
    {
        log_debug(0, "set_device_rocalution()", dev);

        assert(_get_backend_descriptor()->init == false);

        _get_backend_descriptor()->HIP_dev = dev;
    }

    void info_rocalution(void)
    {
        LOG_INFO("rocALUTION ver " << __ROCALUTION_VER_MAJOR << "." << __ROCALUTION_VER_MINOR << "."
                                   << __ROCALUTION_VER_PATCH << __ROCALUTION_VER_PRE << "-"
                                   << TO_STR(__ROCALUTION_VER_TWEAK));

#if defined(__gnu_linux__) || defined(linux) || defined(__linux) || defined(__linux__)

        LOG_VERBOSE_INFO(3, "Compiled for Linux/Unix OS");

#else // Linux

#if defined(__APPLE__)

        LOG_VERBOSE_INFO(3, "Compiled for Mac OS");

#else // Apple

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)

        LOG_VERBOSE_INFO(3, "Compiled for Windows OS");

#else // Win

        // unknown
        LOG_VERBOSE_INFO(3, "Compiled for unknown OS");

#endif // Win

#endif // Apple

#endif // Linux

        info_rocalution(_Backend_Descriptor);
    }

    void info_rocalution(const struct Rocalution_Backend_Descriptor& backend_descriptor)
    {
        if(backend_descriptor.init == true)
        {
            LOG_INFO("rocALUTION platform is initialized");
        }
        else
        {
            LOG_INFO("rocALUTION platform is NOT initialized");
        }

        LOG_INFO("Accelerator backend: " << _rocalution_backend_name[backend_descriptor.backend]);

#ifdef _OPENMP
        LOG_INFO("OpenMP threads: " << backend_descriptor.OpenMP_threads);
#else
        LOG_INFO("No OpenMP support");
#endif

        if(backend_descriptor.disable_accelerator == true)
        {
            LOG_INFO("The accelerator is disabled");
        }

#ifdef SUPPORT_HIP
        if(backend_descriptor.accelerator)
        {
            rocalution_info_hip(backend_descriptor);
        }
        else
        {
            LOG_INFO("HIP is not initialized");
        }
#else
        LOG_VERBOSE_INFO(3, "No HIP support");
#endif

#ifdef SUPPORT_MULTINODE
        LOG_INFO("MPI rank: " << backend_descriptor.rank);

        MPI_Comm comm = MPI_COMM_WORLD;
        int      num_procs;

        int status_init;
        MPI_Initialized(&status_init);

        if(status_init == 1)
        {
            int status = MPI_Comm_size(comm, &num_procs);

            if(status == MPI_SUCCESS)
            {
                LOG_INFO("MPI size: " << num_procs);
            }
            else
            {
                LOG_INFO("MPI is not initialized");
            }
        }
        else
        {
            LOG_INFO("MPI is not initialized");
        }
#else
        LOG_INFO("MPI is not initialized");
#endif
    }

    void set_omp_affinity_rocalution(bool affinity)
    {
        assert(_get_backend_descriptor()->init == false);

        _get_backend_descriptor()->OpenMP_affinity = affinity;
    }

    void set_omp_threshold_rocalution(int threshold)
    {
        assert(_get_backend_descriptor()->init == true);

        _get_backend_descriptor()->OpenMP_threshold = threshold;
    }

    bool _rocalution_available_accelerator(void)
    {
        return _get_backend_descriptor()->accelerator;
    }

    void disable_accelerator_rocalution(bool onoff)
    {
        assert(_get_backend_descriptor()->init == false);

        _get_backend_descriptor()->disable_accelerator = onoff;
    }

    struct Rocalution_Backend_Descriptor* _get_backend_descriptor(void)
    {
        return &_Backend_Descriptor;
    }

    void _set_backend_descriptor(const struct Rocalution_Backend_Descriptor& backend_descriptor)
    {
        *(_get_backend_descriptor()) = backend_descriptor;
    }

    template <typename ValueType>
    AcceleratorVector<ValueType>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor)
    {
        log_debug(0, "_rocalution_init_base_backend_vector()");

        switch(backend_descriptor.backend)
        {
#ifdef SUPPORT_HIP
        // HIP
        case HIP:
            return _rocalution_init_base_hip_vector<ValueType>(backend_descriptor);
#endif

        // LCOV_EXCL_START
        default:
            // No backend supported!
            LOG_INFO("Rocalution was not compiled with "
                     << _rocalution_backend_name[backend_descriptor.backend] << " support");
            LOG_INFO("Building Vector on " << _rocalution_backend_name[backend_descriptor.backend]
                                           << " failed");
            FATAL_ERROR(__FILE__, __LINE__);
            return NULL;
            // LCOV_EXCL_STOP
        }
    }

    template <typename ValueType>
    AcceleratorMatrix<ValueType>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim)
    {
        log_debug(0, "_rocalution_init_base_backend_matrix()", matrix_format, blockdim);

        if(matrix_format != BCSR)
        {
            assert(blockdim == 1);
        }

        switch(backend_descriptor.backend)
        {
#ifdef SUPPORT_HIP
        case HIP:
            return _rocalution_init_base_hip_matrix<ValueType>(
                backend_descriptor, matrix_format, blockdim);
#endif

        // LCOV_EXCL_START
        default:
            LOG_INFO("Rocalution was not compiled with "
                     << _rocalution_backend_name[backend_descriptor.backend] << " support");
            LOG_INFO("Building " << _matrix_format_names[matrix_format] << " Matrix on "
                                 << _rocalution_backend_name[backend_descriptor.backend]
                                 << " failed");

            FATAL_ERROR(__FILE__, __LINE__);
            return NULL;
            // LCOV_EXCL_STOP
        }
    }

    template <typename ValueType>
    HostMatrix<ValueType>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim)
    {
        log_debug(0, "_rocalution_init_base_host_matrix()", matrix_format, blockdim);

        if(matrix_format != BCSR)
        {
            assert(blockdim == 1);
        }

        switch(matrix_format)
        {
        case CSR:
            return new HostMatrixCSR<ValueType>(backend_descriptor);
        case COO:
            return new HostMatrixCOO<ValueType>(backend_descriptor);
        case DIA:
            return new HostMatrixDIA<ValueType>(backend_descriptor);
        case ELL:
            return new HostMatrixELL<ValueType>(backend_descriptor);
        case HYB:
            return new HostMatrixHYB<ValueType>(backend_descriptor);
        case DENSE:
            return new HostMatrixDENSE<ValueType>(backend_descriptor);
        case MCSR:
            return new HostMatrixMCSR<ValueType>(backend_descriptor);
        case BCSR:
            return new HostMatrixBCSR<ValueType>(backend_descriptor, blockdim);
        // LCOV_EXCL_START
        default:
            return NULL;
            // LCOV_EXCL_STOP
        }
    }

    void _rocalution_sync(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_sync();
#endif
        }
    }

    void _rocalution_sync_default(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_sync_default();
#endif
        }
    }

    void _rocalution_sync_interior(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_sync_interior();
#endif
        }
    }

    void _rocalution_sync_ghost(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_sync_ghost();
#endif
        }
    }

    void _rocalution_compute_interior(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_compute_interior();
#endif
        }
    }

    void _rocalution_compute_ghost(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_compute_ghost();
#endif
        }
    }

    void _rocalution_compute_default(void)
    {
        if(_rocalution_available_accelerator() == true)
        {
#ifdef SUPPORT_HIP
            rocalution_hip_compute_default();
#endif
        }
    }

    void _set_omp_backend_threads(const struct Rocalution_Backend_Descriptor& backend_descriptor,
                                  int64_t                                     size)
    {
        // if the threshold is disabled or if the size is not in the threshold limit
        if((backend_descriptor.OpenMP_threshold > 0)
           && (size <= backend_descriptor.OpenMP_threshold) && (size >= 0))
        {
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
        }
        else
        {
#ifdef _OPENMP
            omp_set_num_threads(backend_descriptor.OpenMP_threads);
#endif
        }
    }

    size_t _rocalution_add_obj(class RocalutionObj* ptr)
    {
#ifndef OBJ_TRACKING_OFF

        log_debug(0, "Creating new rocALUTION object, ptr=", ptr);

        Rocalution_Object_Data_Tracking.all_obj.push_back(ptr);

        int id = Rocalution_Object_Data_Tracking.all_obj.size() - 1;

        log_debug(0, "Creating new rocALUTION object, id=", id);

        return id;

#else

        return 0;

#endif
    }

    bool _rocalution_del_obj(class RocalutionObj* ptr, size_t id)
    {
        bool ok = false;

#ifndef OBJ_TRACKING_OFF

        log_debug(0, "Deleting rocALUTION object, ptr=", ptr);

        log_debug(0, "Deleting rocALUTION object, id=", id);

        if(Rocalution_Object_Data_Tracking.all_obj[id] == ptr)
        {
            ok = true;
        }

        Rocalution_Object_Data_Tracking.all_obj[id] = NULL;

        return ok;

#else

        ok = true;

        return ok;

#endif
    }

    void _rocalution_delete_all_obj(void)
    {
#ifndef OBJ_TRACKING_OFF

        log_debug(0, "_rocalution_delete_all_obj()", "* begin");

        for(unsigned int i = 0; i < Rocalution_Object_Data_Tracking.all_obj.size(); ++i)
        {
            if(Rocalution_Object_Data_Tracking.all_obj[i] != NULL)
            {
                Rocalution_Object_Data_Tracking.all_obj[i]->Clear();
            }

            log_debug(
                0, "clearing rocALUTION obj ptr=", Rocalution_Object_Data_Tracking.all_obj[i]);
        }

        Rocalution_Object_Data_Tracking.all_obj.clear();

        log_debug(0, "_rocalution_delete_all_obj()", "* end");
#endif
    }

    bool _rocalution_check_if_any_obj(void)
    {
#ifndef OBJ_TRACKING_OFF

        if(Rocalution_Object_Data_Tracking.all_obj.size() > 0)
        {
            return false;
        }

#endif

        return true;
    }

    std::string get_arch_rocalution(void)
    {
#ifdef SUPPORT_HIP
        if(_get_backend_descriptor()->backend == HIP)
        {
            return rocalution_get_arch_hip();
        }
        else
#endif
        {
            return "cpu";
        }
    }

    template AcceleratorVector<float>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<double>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
#ifdef SUPPORT_COMPLEX
    template AcceleratorVector<std::complex<float>>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<std::complex<double>>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
#endif
    template AcceleratorVector<bool>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<int>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);
    template AcceleratorVector<int64_t>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);

    template AcceleratorMatrix<float>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template AcceleratorMatrix<double>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#ifdef SUPPORT_COMPLEX
    template AcceleratorMatrix<std::complex<float>>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template AcceleratorMatrix<std::complex<double>>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#endif
    template HostMatrix<float>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template HostMatrix<double>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#ifdef SUPPORT_COMPLEX
    template HostMatrix<std::complex<float>>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
    template HostMatrix<std::complex<double>>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim);
#endif

} // namespace rocalution

/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_BACKEND_MANAGER_HPP_
#define ROCALUTION_BACKEND_MANAGER_HPP_

#include "rocalution/export.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace rocalution
{

    template <typename ValueType>
    class AcceleratorVector;
    template <typename ValueType>
    class AcceleratorMatrix;
    template <typename ValueType>
    class HostMatrix;

    /** \ingroup backend_module
  * \struct Rocalution_Backend_Descriptor
  * \brief Rocalution backend descriptor class
  * \details
  * Backend descriptor - keeps information about the hardware - OpenMP (threads); HIP (blocksizes, handles, etc).
  */
    struct Rocalution_Backend_Descriptor
    {
        /** \brief set by initbackend(); */
        bool init;

        /** \brief current backend */
        int backend;
        /** \brief Flag whether to use accelerator */
        bool accelerator;
        /** \brief Flag whether to disable accelerator */
        bool disable_accelerator;

        /** \brief OpenMP threads */
        int OpenMP_threads;
        /** \brief OpenMP threads before ROCALUTION init */
        int OpenMP_def_threads;
        /** \brief OpenMP nested before ROCALUTION init */
        int OpenMP_def_nested;
        /** \brief Host affinity (true-yes/false-no) */
        bool OpenMP_affinity;
        /** \brief Host threshold size */
        int64_t OpenMP_threshold;

        // HIP handle section
        /** \brief rocblas_handle casted in void ** */
        void* ROC_blas_handle;
        /** \brief rocsparse_handle casted in void ** */
        void* ROC_sparse_handle;

        // HIP stream section
        /** \brief HIP dedault stream casted in void ** */
        void* HIP_stream_default;
        /** \brief HIP interior stream casted in void ** */
        void* HIP_stream_interior;
        /** \brief HIP ghost stream casted in void ** */
        void* HIP_stream_ghost;
        /** \brief HIP current stream casted in void ** */
        void* HIP_stream_current;

        /** \brief HIP device identifier */
        int HIP_dev;
        /** \brief HIP warp size */
        int HIP_warp;
        /** \brief HIP block size */
        int HIP_block_size;
        /** \brief HIP max threads */
        int HIP_max_threads;
        /** \brief HIP multiprocessor count */
        int HIP_num_procs;
        /** \brief HIP thread per processor count */
        int HIP_threads_per_proc;

        /** \brief MPI rank/id */
        int rank;

        /** \brief Logging mode */
        int log_mode;
        /** \brief Logging file */
        std::ofstream* log_file;
    };

    // Global backend descriptor
    extern struct Rocalution_Backend_Descriptor _Backend_Descriptor;

    // Host name
    extern const std::string _rocalution_host_name[1];

    // Backend names
    extern const std::string _rocalution_backend_name[2];

    // Backend IDs
    enum _rocalution_backend_id
    {
        None = 0,
        HIP  = 1
    };

    /** \ingroup backend_module
  * \brief Initialize rocALUTION platform
  * \details
  * \p init_rocalution defines a backend descriptor with information about the hardware
  * and its specifications. All objects created after that contain a copy of this
  * descriptor. If the specifications of the global descriptor are changed (e.g. set
  * different number of threads) and new objects are created, only the new objects will
  * use the new configurations.
  *
  * For control, the library provides the following functions
  * - set_device_rocalution() is a unified function to select a specific device. If you
  *   have compiled the library with a backend and for this backend there are several
  *   available devices, you can use this function to select a particular one. This
  *   function has to be called before init_rocalution().
  * - set_omp_threads_rocalution() sets the number of OpenMP threads. This function has
  *   to be called after init_rocalution().
  *
  * @param[in]
  * rank            specifies MPI rank when multi-node environment
  * @param[in]
  * dev_per_node    number of accelerator devices per node, when in multi-GPU environment
  *
  * \par Example
  * \code{.cpp}
  *   #include <rocalution/rocalution.hpp>
  *
  *   using namespace rocalution;
  *
  *   int main(int argc, char* argv[])
  *   {
  *       init_rocalution();
  *
  *       // ...
  *
  *       stop_rocalution();
  *
  *       return 0;
  *   }
  * \endcode
  */
    ROCALUTION_EXPORT
    int init_rocalution(int rank = -1, int dev_per_node = 1);

    /** \ingroup backend_module
  * \brief Shutdown rocALUTION platform
  * \details
  * \p stop_rocalution shuts down the rocALUTION platform.
  */
    ROCALUTION_EXPORT
    int stop_rocalution(void);

    /** \ingroup backend_module
  * \brief Set the accelerator device
  * \details
  * \p set_device_rocalution lets the user select the accelerator device that is supposed
  * to be used for the computation.
  *
  * @param[in]
  * dev     accelerator device ID for computation
  */
    ROCALUTION_EXPORT
    void set_device_rocalution(int dev);

    /** \ingroup backend_module
  * \brief Set number of OpenMP threads
  * \details
  * The number of threads which rocALUTION will use can be set with
  * \p set_omp_threads_rocalution or by the global OpenMP environment variable (for
  * Unix-like OS this is \p OMP_NUM_THREADS). During the initialization phase, the
  * library provides affinity thread-core mapping:
  * - If the number of cores (including SMT cores) is greater or equal than two times the
  *   number of threads, then all the threads can occupy every second core ID (e.g. 0, 2,
  *   4, \f$\ldots\f$). This is to avoid having two threads working on the same physical
  *   core, when SMT is enabled.
  * - If the number of threads is less or equal to the number of cores (including SMT),
  *   and the previous clause is false, then the threads can occupy every core ID (e.g.
  *   0, 1, 2, 3, \f$\ldots\f$).
  * - If non of the above criteria is matched, then the default thread-core mapping is
  *   used (typically set by the OS).
  *
  * \note
  * The thread-core mapping is available only for Unix-like OS.
  *
  * \note
  * The user can disable the thread affinity by calling set_omp_affinity_rocalution(),
  * before initializing the library (i.e. before init_rocalution()).
  *
  * @param[in]
  * nthreads    number of OpenMP threads
  */
    ROCALUTION_EXPORT
    void set_omp_threads_rocalution(int nthreads);

    /** \ingroup backend_module
  * \brief Enable/disable OpenMP host affinity
  * \details
  * \p set_omp_affinity_rocalution enables / disables OpenMP host affinity.
  *
  * @param[in]
  * affinity    boolean to turn on/off OpenMP host affinity
  */
    ROCALUTION_EXPORT
    void set_omp_affinity_rocalution(bool affinity);

    /** \ingroup backend_module
  * \brief Set OpenMP threshold size
  * \details
  * Whenever you want to work on a small problem, you might observe that the OpenMP host
  * backend is (slightly) slower than using no OpenMP. This is mainly attributed to the
  * small amount of work, which every thread should perform and the large overhead of
  * forking/joining threads. This can be avoid by the OpenMP threshold size parameter in
  * rocALUTION. The default threshold is set to 10000, which means that all matrices
  * under (and equal) this size will use only one thread (disregarding the number of
  * OpenMP threads set in the system). The threshold can be modified with
  * \p set_omp_threshold_rocalution.
  *
  * @param[in]
  * threshold   OpenMP threshold size
  */
    ROCALUTION_EXPORT
    void set_omp_threshold_rocalution(int threshold);

    /** \ingroup backend_module
  * \brief Print info about rocALUTION
  * \details
  * \p info_rocalution prints information about the rocALUTION platform
  */
    ROCALUTION_EXPORT
    void info_rocalution(void);

    /** \ingroup backend_module
  * \brief Print info about specific rocALUTION backend descriptor
  * \details
  * \p info_rocalution prints information about the rocALUTION platform of the specific
  * backend descriptor.
  *
  * @param[in]
  * backend_descriptor  rocALUTION backend descriptor
  */
    ROCALUTION_EXPORT
    void info_rocalution(const struct Rocalution_Backend_Descriptor& backend_descriptor);

    /** \ingroup backend_module
  * \brief Disable/Enable the accelerator
  * \details
  * If you want to disable the accelerator (without re-compiling the code), you need to
  * call \p disable_accelerator_rocalution before init_rocalution().
  *
  * @param[in]
  * onoff   boolean to turn on/off the accelerator
  */
    ROCALUTION_EXPORT
    void disable_accelerator_rocalution(bool onoff = true);

    // Return true if any accelerator is available
    bool _rocalution_available_accelerator(void);

    // Return backend descriptor
    struct Rocalution_Backend_Descriptor* _get_backend_descriptor(void);

    // Set backend descriptor
    void _set_backend_descriptor(const struct Rocalution_Backend_Descriptor& backend_descriptor);

    // Set the OMP threads based on the size threshold
    void _set_omp_backend_threads(const struct Rocalution_Backend_Descriptor& backend_descriptor,
                                  int64_t                                     size);

    // Build (and return) a vector on the selected in the descriptor accelerator
    template <typename ValueType>
    AcceleratorVector<ValueType>* _rocalution_init_base_backend_vector(
        const struct Rocalution_Backend_Descriptor& backend_descriptor);

    // Build (and return) a matrix on the host
    template <typename ValueType>
    HostMatrix<ValueType>* _rocalution_init_base_host_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim = 1);

    // Build (and return) a matrix on the selected in the descriptor accelerator
    template <typename ValueType>
    AcceleratorMatrix<ValueType>* _rocalution_init_base_backend_matrix(
        const struct Rocalution_Backend_Descriptor& backend_descriptor,
        unsigned int                                matrix_format,
        int                                         blockdim = 1);

    /** \ingroup backend_module
  * \brief Sync rocALUTION
  * \details
  * \p _rocalution_sync blocks the host until all active asynchronous transfers are completed (this is a global sync).
  */
    ROCALUTION_EXPORT
    void _rocalution_sync(void);

    /** \ingroup backend_module
  * \brief Sync rocALUTION default
  * \details
  * \p _rocalution_sync_default blocks the host until all active asynchronous transfers in default compute mode are completed.
  */
    ROCALUTION_EXPORT
    void _rocalution_sync_default(void);

    /** \ingroup backend_module
  * \brief Sync rocALUTION interior
  * \details
  * \p _rocalution_sync_interior blocks the host until all active asynchronous transfers in interior compute mode are completed.
  */
    ROCALUTION_EXPORT
    void _rocalution_sync_interior(void);

    /** \ingroup backend_module
  * \brief Sync rocALUTION ghost
  * \details
  * \p _rocalution_sync_ghost blocks the host until all active asynchronous transfers in ghost compute mode are completed.
  */
    ROCALUTION_EXPORT
    void _rocalution_sync_ghost(void);

    /** \ingroup backend_module
  * \brief Set compute mode to interior
  * \details
  * \p _rocalution_compute_interior sets sync compute mode to interior.
  */
    ROCALUTION_EXPORT
    void _rocalution_compute_interior(void);

    /** \ingroup backend_module
  * \brief Set compute mode to ghost
  * \details
  * \p _rocalution_compute_ghost sets sync compute mode to ghost.
  */
    ROCALUTION_EXPORT
    void _rocalution_compute_ghost(void);

    /** \ingroup backend_module
  * \brief Set compute mode to default
  * \details
  * \p _rocalution_compute_default sets sync compute mode to default.
  */
    ROCALUTION_EXPORT
    void _rocalution_compute_default(void);

    /** \ingroup backend_module
  * \brief get rocALUTION backend architecture
  * \details
  * \p get_arch_rocalution returns the name of rocalution's backend architecture
  */
    ROCALUTION_EXPORT
    std::string get_arch_rocalution(void);

    size_t _rocalution_add_obj(class RocalutionObj* ptr);
    bool   _rocalution_del_obj(class RocalutionObj* ptr, size_t id);
    void   _rocalution_delete_all_obj(void);
    bool   _rocalution_check_if_any_obj(void);

} // namespace rocalution

#endif // ROCALUTION_BACKEND_MANAGER_HPP_

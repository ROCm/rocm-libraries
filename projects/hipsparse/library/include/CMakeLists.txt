# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GenerateExportHeader)

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse-version.h.in"
    "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h"
    @ONLY
)

# Copy public header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse.h"
    "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse.h"
    COPYONLY
)

set(HEADER_FILES
    # Level1
    internal/level1/hipsparse_axpyi.h
    internal/level1/hipsparse_dotci.h
    internal/level1/hipsparse_doti.h
    internal/level1/hipsparse_gthr.h
    internal/level1/hipsparse_gthrz.h
    internal/level1/hipsparse_roti.h
    internal/level1/hipsparse_sctr.h
    # Level2
    internal/level2/hipsparse_bsrmv.h
    internal/level2/hipsparse_bsrsv.h
    internal/level2/hipsparse_bsrxmv.h
    internal/level2/hipsparse_csrmv.h
    internal/level2/hipsparse_csrsv.h
    internal/level2/hipsparse_gemvi.h
    internal/level2/hipsparse_hybmv.h
    # Level3
    internal/level3/hipsparse_bsrmm.h
    internal/level3/hipsparse_bsrsm.h
    internal/level3/hipsparse_csrmm.h
    internal/level3/hipsparse_csrsm.h
    internal/level3/hipsparse_gemmi.h
    # Extra
    internal/extra/hipsparse_csrgeam.h
    internal/extra/hipsparse_csrgemm.h
    # Precond
    internal/precond/hipsparse_bsric0.h
    internal/precond/hipsparse_bsrilu0.h
    internal/precond/hipsparse_csric0.h
    internal/precond/hipsparse_csrilu0.h
    internal/precond/hipsparse_gpsv_interleaved_batch.h
    internal/precond/hipsparse_gtsv_interleaved_batch.h
    internal/precond/hipsparse_gtsv_nopivot.h
    internal/precond/hipsparse_gtsv_strided_batch.h
    internal/precond/hipsparse_gtsv.h
    # Reorder
    internal/reorder/hipsparse_csrcolor.h
    # Conversion
    internal/conversion/hipsparse_bsr2csr.h
    internal/conversion/hipsparse_coo2csr.h
    internal/conversion/hipsparse_coosort.h
    internal/conversion/hipsparse_create_identity_permutation.h
    internal/conversion/hipsparse_csc2dense.h
    internal/conversion/hipsparse_cscsort.h
    internal/conversion/hipsparse_csr2bsr.h
    internal/conversion/hipsparse_csr2coo.h
    internal/conversion/hipsparse_csr2csc.h
    internal/conversion/hipsparse_csr2csr_compress.h
    internal/conversion/hipsparse_csr2csru.h
    internal/conversion/hipsparse_csr2dense.h
    internal/conversion/hipsparse_csr2gebsr.h
    internal/conversion/hipsparse_csr2hyb.h
    internal/conversion/hipsparse_csrsort.h
    internal/conversion/hipsparse_csru2csr.h
    internal/conversion/hipsparse_dense2csc.h
    internal/conversion/hipsparse_dense2csr.h
    internal/conversion/hipsparse_gebsr2csr.h
    internal/conversion/hipsparse_gebsr2gebsc.h
    internal/conversion/hipsparse_gebsr2gebsr.h
    internal/conversion/hipsparse_hyb2csr.h
    internal/conversion/hipsparse_nnz_compress.h
    internal/conversion/hipsparse_nnz.h
    internal/conversion/hipsparse_prune_csr2csr_by_percentage.h
    internal/conversion/hipsparse_prune_csr2csr.h
    internal/conversion/hipsparse_prune_dense2csr_by_percentage.h
    internal/conversion/hipsparse_prune_dense2csr.h
    # Generic
    internal/generic/hipsparse_axpby.h
    internal/generic/hipsparse_dense2sparse.h
    internal/generic/hipsparse_gather.h
    internal/generic/hipsparse_rot.h
    internal/generic/hipsparse_scatter.h
    internal/generic/hipsparse_sddmm.h
    internal/generic/hipsparse_sparse2dense.h
    internal/generic/hipsparse_spgemm_reuse.h
    internal/generic/hipsparse_spgemm.h
    internal/generic/hipsparse_spmm.h
    internal/generic/hipsparse_spmv.h
    internal/generic/hipsparse_spsm.h
    internal/generic/hipsparse_spsv.h
    internal/generic/hipsparse_spvv.h
    # Auxiliary
    hipsparse-bfloat16.h
    hipsparse-float16.h
    hipsparse-types.h
    hipsparse-auxiliary.h
    hipsparse-generic-types.h
    hipsparse-generic-auxiliary.h
)

foreach(file_i ${HEADER_FILES})
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/${file_i}"
        "${PROJECT_BINARY_DIR}/include/hipsparse/${file_i}"
        COPYONLY
    )
endforeach()

generate_export_header(hipsparse
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-export.h"
)

target_include_directories(hipsparse
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsparse>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipsparse>
)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# HipBLASLtCodegenInstall.cmake — install the *codegen subset* of TensileLite so
# downstream consumers (e.g. hipSPARSELt) can call hipblaslt_create_device_library()
# against a binary-only hipBLASLt install with no hipBLASLt source tree present.
#
# This is NOT the full TensileLite Python tree (that 90 MB tree, including the
# tuner/benchmark code and the 84 MB Tensile/CustomKernels/ assembly, ships only
# under HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS for artifact-based testing).
# Here we ship only the static import closure of
#   python -m Tensile.TensileCreateLibrary   (device-library codegen)
#   Tensile/bin/TensileLogic                 (the pre-build --check-all gate)
# plus the runtime data those entry points read, and the compiled rocisa
# extension they import. Total ~3.7 MB.
#
# The pinned file list below is the exact closure produced by
# .cmake-work/closure_trace.py (83 reached modules) plus the two package
# __init__.py that exist in the source tree but are only reached transitively
# (Tensile/Components/Subtile, Tensile/Toolchain) — both required so the installed
# package hierarchy is importable. Regenerate with closure_trace.py if the codegen
# entry points gain or drop first-party imports.

include_guard(GLOBAL)

# hipblaslt_install_codegen_subset(<tensilelite-source-dir> [COMPONENT <name>])
#   <tensilelite-source-dir>  the in-tree tensilelite/ dir holding Tensile/ + rocisa/
# Installs into ${CMAKE_INSTALL_DATADIR}/hipblaslt/codegen/ :
#   Tensile/<closure .py>, Tensile/Source/, Tensile/bin/TensileLogic,
#   Tensile/TensileLogic/known_bugs.yaml, rocisa/{__init__.py,_rocisa.so,stinkytofu}
function(hipblaslt_install_codegen_subset _src)
    set(_opts "")
    set(_one COMPONENT)
    set(_multi "")
    cmake_parse_arguments(_cg "${_opts}" "${_one}" "${_multi}" ${ARGN})
    if(NOT _cg_COMPONENT)
        set(_cg_COMPONENT runtime)
    endif()

    set(_dest "${CMAKE_INSTALL_DATADIR}/hipblaslt/codegen")

    # --- Pinned Tensile codegen import closure (85 .py) ---
    set(_codegen_py
        Tensile/Activation.py
        Tensile/AsmAddressCalculation.py
        Tensile/AsmMemoryHelpers.py
        Tensile/AsmMemoryInstruction.py
        Tensile/AsmStoreState.py
        Tensile/Common/Architectures.py
        Tensile/Common/Capabilities.py
        Tensile/Common/Constants.py
        Tensile/Common/DataType.py
        Tensile/Common/GlobalParameters.py
        Tensile/Common/Parallel.py
        Tensile/Common/RegisterPool.py
        Tensile/Common/RequiredParameters.py
        Tensile/Common/TimingInstrumentation.py
        Tensile/Common/Types.py
        Tensile/Common/Utilities.py
        Tensile/Common/ValidParameters.py
        Tensile/Common/__init__.py
        Tensile/Component.py
        Tensile/Components/CMSValidator.py
        Tensile/Components/CustomSchedule.py
        Tensile/Components/NonTemporal.py
        Tensile/Components/Signature.py
        Tensile/Components/Subtile/InstructionEmitter.py
        Tensile/Components/Subtile/InstructionScheduler.py
        Tensile/Components/Subtile/Kernel.py
        Tensile/Components/Subtile/LogicalScheduler.py
        Tensile/Components/Subtile/SubtileGREmit.py
        Tensile/Components/Subtile/SubtileGeometry.py
        Tensile/Components/Subtile/SubtileLREmit.py
        Tensile/Components/Subtile/SubtileScaleEmit.py
        Tensile/Components/Subtile/__init__.py
        Tensile/Components/TensorDataMover.py
        Tensile/Components/WorkGroupMappingAlgos.py
        Tensile/Components/__init__.py
        Tensile/Contractions.py
        Tensile/CustomKernels.py
        Tensile/CustomYamlLoader.py
        Tensile/Hardware.py
        Tensile/KernelHelperNaming.py
        Tensile/KernelWriter.py
        Tensile/KernelWriterActivationEnumHeader.py
        Tensile/KernelWriterActivationFunction.py
        Tensile/KernelWriterAssembly.py
        Tensile/KernelWriterBase.py
        Tensile/KernelWriterBetaOnly.py
        Tensile/KernelWriterConversion.py
        Tensile/KernelWriterModules.py
        Tensile/KernelWriterReduction.py
        Tensile/LibraryIO.py
        Tensile/Properties.py
        Tensile/SolutionLibrary.py
        Tensile/SolutionStructs/LdsPadding.py
        Tensile/SolutionStructs/Naming.py
        Tensile/SolutionStructs/Problem.py
        Tensile/SolutionStructs/Solution.py
        Tensile/SolutionStructs/Utilities.py
        Tensile/SolutionStructs/Validators/MXScaleFormat.py
        Tensile/SolutionStructs/Validators/MatrixInstruction.py
        Tensile/SolutionStructs/Validators/WorkGroup.py
        Tensile/SolutionStructs/__init__.py
        Tensile/TensileCreateLibrary/ParseArguments.py
        Tensile/TensileCreateLibrary/Run.py
        Tensile/TensileCreateLibrary/__init__.py
        Tensile/TensileCreateLibrary/__main__.py
        Tensile/TensileLogic/HandleCustomKernel.py
        Tensile/TensileLogic/KnownBugs.py
        Tensile/TensileLogic/ParseArguments.py
        Tensile/TensileLogic/Run.py
        Tensile/TensileLogic/ValidChipId.py
        Tensile/TensileLogic/ValidMatrixInstruction.py
        Tensile/TensileLogic/ValidWorkGroup.py
        Tensile/TensileLogic/ValidWorkGroupMappingXCC.py
        Tensile/TensileLogic/__init__.py
        Tensile/Toolchain/Assembly.py
        Tensile/Toolchain/Component.py
        Tensile/Toolchain/HelperKernelCache.py
        Tensile/Toolchain/Source.py
        Tensile/Toolchain/Validators.py
        Tensile/Toolchain/__init__.py
        Tensile/Utilities/Decorators/Profile.py
        Tensile/Utilities/Decorators/Shared.py
        Tensile/Utilities/Decorators/Timing.py
        Tensile/__init__.py
        Tensile/verify_stinky_comment_vs_elf_text.py
    )

    foreach(_rel IN LISTS _codegen_py)
        get_filename_component(_reldir "${_rel}" DIRECTORY)
        install(
            FILES "${_src}/${_rel}"
            DESTINATION "${_dest}/${_reldir}"
            COMPONENT ${_cg_COMPONENT}
        )
    endforeach()

    # --- Runtime data the codegen reads __file__-relative ---
    # Tensile/Source/: kernel headers copied into the output by Run.py.
    install(
        DIRECTORY "${_src}/Tensile/Source/"
        DESTINATION "${_dest}/Tensile/Source"
        COMPONENT ${_cg_COMPONENT}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )
    # known_bugs.yaml: consumed by the TensileLogic --check-all gate (not a .py,
    # so not part of the import closure above).
    install(
        FILES "${_src}/Tensile/TensileLogic/known_bugs.yaml"
        DESTINATION "${_dest}/Tensile/TensileLogic"
        COMPONENT ${_cg_COMPONENT}
    )
    # The TensileLogic gate is invoked as a script (keep +x).
    install(
        PROGRAMS "${_src}/Tensile/bin/TensileLogic"
        DESTINATION "${_dest}/Tensile/bin"
        COMPONENT ${_cg_COMPONENT}
    )

    # D1 safety net: the third-party pip deps of the codegen import closure. Not
    # vendored (unlike rocisa) — the consumer's interpreter must have them. Shipped
    # as a manifest so a TheRock/packaging step can pip-install if needed.
    install(
        FILES "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/codegen-requirements.txt"
        DESTINATION "${_dest}"
        RENAME requirements.txt
        COMPONENT ${_cg_COMPONENT}
    )

    # --- Vendored rocisa (D2: not a package; rides along as a codegen impl
    # detail). rocisa/__init__.py does `from . import _rocisa`, so the compiled
    # extension must sit inside the rocisa/ package dir. _build_info.py is NOT
    # shipped, so the package's source-staleness check self-skips on installs.
    install(
        FILES "${_src}/rocisa/rocisa/__init__.py"
        DESTINATION "${_dest}/rocisa"
        COMPONENT ${_cg_COMPONENT}
    )
    if(TARGET _rocisa)
        install(
            TARGETS _rocisa
            DESTINATION "${_dest}/rocisa"
            COMPONENT ${_cg_COMPONENT}
        )
    endif()
    # stinkytofu is linked PRIVATE into _rocisa. When it is a STATIC library it is
    # absorbed into _rocisa.so (self-contained, nothing extra to ship); only when
    # it is a separate SHARED library must we co-install it next to _rocisa so the
    # binding resolves at runtime. Guarding on the target TYPE also avoids a
    # generate-time error from TARGET_SONAME_FILE on a non-shared target.
    if(TARGET stinkytofu AND NOT WIN32)
        get_target_property(_st_type stinkytofu TYPE)
        if(_st_type STREQUAL "SHARED_LIBRARY")
            install(FILES $<TARGET_FILE:stinkytofu>
                DESTINATION "${_dest}/rocisa"
                COMPONENT ${_cg_COMPONENT}
            )
            install(FILES $<TARGET_SONAME_FILE:stinkytofu>
                DESTINATION "${_dest}/rocisa"
                COMPONENT ${_cg_COMPONENT}
            )
        endif()
    endif()
endfunction()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

function(hipblaslt_install_codegen_subset _src)
    set(_opts "")
    set(_one COMPONENT)
    set(_multi "")
    cmake_parse_arguments(_cg "${_opts}" "${_one}" "${_multi}" ${ARGN})
    if(NOT _cg_COMPONENT)
        set(_cg_COMPONENT runtime)
    endif()

    set(_dest "${CMAKE_INSTALL_DATADIR}/hipblaslt/codegen")

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

    install(
        DIRECTORY "${_src}/Tensile/Components/"
        DESTINATION "${_dest}/Tensile/Components"
        COMPONENT ${_cg_COMPONENT}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )

    install(
        DIRECTORY "${_src}/Tensile/Source/"
        DESTINATION "${_dest}/Tensile/Source"
        COMPONENT ${_cg_COMPONENT}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )
    install(
        FILES "${_src}/Tensile/TensileLogic/known_bugs.yaml"
        DESTINATION "${_dest}/Tensile/TensileLogic"
        COMPONENT ${_cg_COMPONENT}
    )
    install(
        PROGRAMS "${_src}/Tensile/bin/TensileLogic"
        DESTINATION "${_dest}/Tensile/bin"
        COMPONENT ${_cg_COMPONENT}
    )

    install(
        FILES "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/codegen-requirements.txt"
        DESTINATION "${_dest}"
        RENAME requirements.txt
        COMPONENT ${_cg_COMPONENT}
    )

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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hip_kernel_provider_tests
    main.cpp
    core/TestContainer.cpp
    core/TestHandle.cpp
    core/TestContext.cpp
    core/TestUtils.cpp
    core/TestPluginPublic.cpp
    compilation/TestProgramAndKernel.cpp
    compilation/TestKernelCompileOptions.cpp
    compilation/TestModuleCache.cpp
    device/TestScopedDevice.cpp
    utilities/TestDigest.cpp
)

# Add the dummy kernel for running the tests
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelEmbedding.cmake")
add_kernels_for_embedding(TARGET hip_kernel_provider_tests FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vector_add.cpp)

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # The kernel sources the ingestor engine's unit descriptor set names. Each key is the
    # `source_file` its descriptor authors, which is the string the runtime looks up.
    # hkp_verify_embedded_sources() compares this list against the staged descriptors.
    set(_ingestor_kernel_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/test_descriptors/unit/pointwise/kernels")
    # The pointwise_model pack authors its own kernel beside its own descriptors. A packed
    # root is validated whole: hkp_verify_embedded_sources() resolves a descriptor's authored
    # location from the pack registry and requires the embedded source to come from inside
    # that root, so this pack cannot share the unit set's file -- nor its key, which already
    # belongs to another descriptor.
    # The STAGED copy, not the authored one: hkp_verify_embedded_sources() resolves this
    # pack's authored location through the pack registry, which names the staging root, and
    # requires the embedded file to have come from inside it. HkpPackaging.cmake puts the
    # kernel there at configure time, before this directory is added.
    set(_model_kernel_dir "${HIPKERNELPROVIDER_POINTWISE_MODEL_STAGE_DIR}/kernels")
    add_kernels_for_embedding(
        TARGET hip_kernel_provider_tests
        FILES "${_ingestor_kernel_dir}/PointwiseAdd.cpp"
              "${_ingestor_kernel_dir}/PointwiseMul.cpp"
              "${_ingestor_kernel_dir}/PointwiseSub.cpp"
              "${_model_kernel_dir}/PointwiseModelAdd.cpp"
        KEYS kernels/PointwiseAdd.cpp
             kernels/PointwiseMul.cpp
             kernels/PointwiseSub.cpp
             kernels/PointwiseModelAdd.cpp
    )
endif()
embed_kernel_sources(
    TARGET hip_kernel_provider_tests
    OUTPUT_SRCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    OUTPUT_SRCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.hpp
    OUTPUT_INCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp
    OUTPUT_INCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.hpp
)
set(HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES} TARGET_DIRECTORY hip_kernel_provider_tests PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_tests PRIVATE "${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES}")

target_compile_options(hip_kernel_provider_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hip_kernel_provider_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                   hip_kernel_provider_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find files without having to use a relative path.
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_CURRENT_BINARY_DIR}/engines)

if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA unit tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        add_subdirectory(engines/asm_sdpa_engine)
    endif()
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    add_subdirectory(engines/hip_mlops_engine)
endif()

if(ENABLE_HIP_FLASH2_ENGINE)
    add_subdirectory(engines/hip_flash2_engine)
endif()
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # The descriptor census, in its own binary.
    #
    # A census suite is not an ordinary unit: it states the inventory of ONE pack
    # target's shard for ONE architecture, both of which the caller supplies through the
    # environment. Left in hip_kernel_provider_tests those suites run in every bare
    # invocation of it with neither supplied -- against the unit shard on a build that
    # may not even pack their bundle -- where they can only fail or skip, and say nothing
    # about the descriptors either way.
    #
    # Same main() as the ordinary binary, so the census listener, the cache redirect and
    # the compiled-in descriptor root behave identically, and the same embedded-source
    # table, so TestPointwisePacks resolves the keys its descriptors name.
    add_executable(
        hip_kernel_provider_census_tests
        main.cpp
        ${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES}
    )

    set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES}
        TARGET_DIRECTORY hip_kernel_provider_census_tests
        PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

    target_compile_options(hip_kernel_provider_census_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

    target_link_libraries(
        hip_kernel_provider_census_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                       hip_kernel_provider_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
    )

    target_include_directories(hip_kernel_provider_census_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_CURRENT_BINARY_DIR}/engines)

    add_subdirectory(engines/kernel_ingestor_engine)

    # descriptorSearchDirectory()'s tests assert which of its three sources answered, and
    # the only way to tell branch 2 or 3 from a failed lookup is that both end in this
    # fixed offset. Same value the provider is compiled with, so the test cannot drift
    # into asserting a path the provider never builds.
    #
    # One define per descriptor set. A single define cannot serve both: the discovery set is
    # the whole unit root, and the kpack cases need the one subtree under it that carries an
    # archive.
    #
    # Binary-relative, never an absolute build path. The offset holds in the build tree and
    # the install tree alike, so these cases read the descriptors beside whichever copy of
    # the binary is running instead of reaching back into the tree that configured it -- an
    # installed run would otherwise find nothing, enumerate nothing, and pass.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        # The unit root, the first tree TestShippedDescriptorSets validates. The unit root is
        # the one this binary stages, so pointing anywhere else makes every case there
        # vacuous, which is the exact failure the first test in that file exists to catch.
        # The production tree it also validates is defined below, where this build packs one.
        HIPDNN_TEST_DESCRIPTOR_DIR="${HIPKERNELPROVIDER_UNIT_BUILD_DIR}"
        HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR="${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
        HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}"
        # The shipped packs, which the pack census cases assert against. Separate from the
        # unit root above because the two are reached by different mechanisms: the unit root
        # replaces discovery and this one extends it.
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR="${HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR}"
        HIPKERNELPROVIDER_UNIT_KPACK_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}/${HIPKERNELPROVIDER_TEST_SET_SHARED}")

    # The same descriptor defines the ordinary binary carries, minus the kpack reldir, so
    # the shared main() resolves its root identically in both. The archive cases that read
    # the kpack reldir stay in the ordinary binary, and a define no source here consumes
    # would only invite a census case to start depending on one.
    target_compile_definitions(hip_kernel_provider_census_tests PRIVATE
        HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR="${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
        HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}")

    # The kpack adapter's tests. Gated with the sources they exercise, and kept together
    # here -- including the dispatch ones, which sit under engines/ -- so they sit between
    # the two defines they read. Those are HIPKERNELPROVIDER_UNIT_KPACK_RELDIR above, under
    # which the device cases find the archive this build packed for the local arch, and
    # HIPKERNELPROVIDER_TEST_KPACK_ARCHIVE_RELPATH below, which the parse-level cases open
    # because they need a real container rather than a matching device.
    target_sources(hip_kernel_provider_tests PRIVATE
        compilation/TestKpackArchive.cpp
        compilation/TestKpackKernelLoader.cpp
        compilation/TestKpackModuleCache.cpp
        engines/kernel_ingestor_engine/packs/TestPointwiseKpackDispatch.cpp
        ingestor/TestPackedDescriptorLoad.cpp
    )

    # A real archive to open, so the arch-mismatch and missing-toc_key cases meet the
    # actual reader rather than a hand-forged header. Vendored from rocm-kpack's own
    # test assets: the reader is now the platform's shared library, so there is no
    # fetched source tree to take it from, and a 518-byte container is cheaper to carry
    # than a build-time dependency on one.
    #
    # Staged into the test descriptor tree and named by a path relative to this binary,
    # as the descriptor sets are, so the cases open the archive that ships with whichever
    # copy of this binary runs. That binary also runs from a packaged artifact carrying no
    # checkout, where a relative path is the only form that resolves. install(DIRECTORY)
    # on that tree ships the archive with the descriptors.
    set(_hkp_test_kpack_subdir "test_kpack_binary")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_zstd.kpack"
        "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}/${_hkp_test_kpack_subdir}/test_zstd.kpack"
        COPYONLY)
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPKERNELPROVIDER_TEST_KPACK_ARCHIVE_RELPATH="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${_hkp_test_kpack_subdir}/test_zstd.kpack")

    # This binary reads the per-arch descriptors and archives the packing roots stage under
    # test_arch_content/, and each packing target is reachable only through ALL.
    # Building this binary by name -- what `ninja check` does -- otherwise runs the cases
    # against a directory that does not exist yet, and they skip rather than fail.
    # Each descriptor set has its own staging target. Name each one individually.
    foreach(_hkp_set hkp_packaging_unit hkp_packaging_unit_shared
                     hkp_packaging_${HIPKERNELPROVIDER_POINTWISE_MODEL_SET}
                     hkp_pointwise_model_artifact)
        if(TARGET ${_hkp_set})
            add_dependencies(hip_kernel_provider_tests ${_hkp_set})
            add_dependencies(hip_kernel_provider_census_tests ${_hkp_set})
        endif()
    endforeach()

    # The production shard, wired only where a production root holds descriptors this
    # build packs for. The census binary reads its shard, and TestShippedDescriptorSets in
    # the ordinary binary validates the tree whole, through the same binary-relative offset
    # the integration binary loads it by. Absent, there is nothing to depend on, no census
    # entry is registered against it, and that suite's production instance skips.
    if(TARGET hkp_packaging_product)
        add_dependencies(hip_kernel_provider_census_tests hkp_packaging_product)
        add_dependencies(hip_kernel_provider_tests hkp_packaging_product)
        target_compile_definitions(hip_kernel_provider_tests PRIVATE
            HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}")
    endif()

    # The parent of the unit descriptor sets, the only root whose embedded sources this
    # binary carries.
    # A walk of a parent reaches every subtree staged under it, so a further pack
    # there comes under the check with no edit here.
    hkp_verify_embedded_sources(
        TARGET hip_kernel_provider_tests
        STAGED_DESCRIPTOR_ROOTS ${HIPKERNELPROVIDER_UNIT_BUILD_DIR}
        # pointwise_model is packed into this root too. Its kernel is the embedded
        # PointwiseModelAdd.cpp registered above, which is why it is staged here rather than
        # beside the integration descriptors: an embedded_source descriptor needs a binary
        # that carries the source, and only this one does.
        PACK_NAMES unit unit_shared ${HIPKERNELPROVIDER_POINTWISE_MODEL_SET}
    )

    # One census entry per architecture the `unit` pack was wired for, each running this
    # binary against that pack's own shard. A census entry names one shard and the listener
    # requires every case in the suite to pass without skipping, so only a single-pack suite
    # qualifies: every TestPointwisePacks case reads loadedSet("hipkernel:Pointwise"), which
    # the `unit` root holds, while TestConvFwdPack reaches across unit and unit_shared and
    # is censused nowhere.
    #
    # EXPECTED_CASES pins the suite's case-name set: the listener builds its obligations
    # from the suite itself, so a dropped case takes its obligation with it and the pin is
    # what notices. Each entry is registered with three controls -- the suite with every
    # case filtered out, a root that does not exist, and a pin naming an unregistered case
    # -- each matched against the diagnostic it must provoke.
    hkp_register_census_tests(
        TARGET hip_kernel_provider_census_tests
        PACK_NAME unit
        SUITES TestPointwisePacks
        EXPECTED_CASES
            EachPackShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes
            EveryKernelNamesItsPacksEmbeddedSource
            EveryEmbeddedSourceKeyResolvesInTheCompiledInTable
            EveryPackNamesTheArchitectureItWasPackedFor
            EveryPackSharesTheEngineDispatchAndAllButOneMatcher
            ExposesBlockSizeAsAKnobAndDtypeAsInternal
            MatchersCoverBothScopes
            SubtractsInTheRightDirection
    )

    # Census for the gfx950 dense-attention production pack. Reads only the product
    # shard, so it is eligible for census registration.
    #
    # The predicate and ARCHES narrow independently -- whether `product` carries THIS
    # bundle at all, and which of its wired architectures the bundle emits for. Either
    # alone lets a census address a shard its suite has nothing to say about.
    #
    # The Pointwise call above needs neither: its fixtures are the unit root itself, and
    # they cover every architecture it is wired for.
    hkp_gfx950_attention_dense_available(_hkp_gfx950_attention_dense)
    if(_hkp_gfx950_attention_dense)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx950
            SUITES TestGfx950AttentionDensePacks
            EXPECTED_CASES
                LoadsExactlyTheEmittedPacks
                LoadsExactlyTheEmittedKernels
                EveryLoadedDescriptorCoversTheRequestedArch
                EveryKernelDeclaresTheEmittedRuntimeSourceKind
                DeclaresTheConfiguredSdkVersion
                ExposesTheConfiguredKnobs
                RanksThroughItsRegisteredScoreSymbol
                CarriesNoGraphScopedMatcher
                CarriesExactlyOneSharedKernelScopedMatcher
        )
    else()
        message(STATUS
            "hkp: this configuration does not ship hipkernel:Gfx950AttentionDense for "
            "gfx950, so its census is not registered.")
    endif()

    # Census for the gfx950 forward-convolution production pack; same shard, same
    # predicate shape and the same ARCHES narrowing as the dense-attention call above.
    hkp_gfx950_conv_fwd_available(_hkp_gfx950_conv_fwd)
    if(_hkp_gfx950_conv_fwd)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx950
            SUITES TestGfx950ConvFwdPacks
            EXPECTED_CASES
                LoadsExactlyTheEmittedPacks
                LoadsExactlyTheEmittedKernels
                EveryLoadedDescriptorCoversTheRequestedArch
                EveryKernelDeclaresTheEmittedRuntimeSourceKind
                DeclaresTheConfiguredSdkVersion
                ExposesTheConfiguredKnobs
                RanksThroughItsRegisteredScoreSymbol
                CarriesNoGraphScopedMatcher
                CarriesExactlyOneSharedKernelScopedMatcher
        )
    else()
        message(STATUS
            "hkp: this configuration does not ship hipkernel:Gfx950ConvFwd for "
            "gfx950, so its census is not registered.")
    endif()
endif()

clang_tidy_check(hip_kernel_provider_tests)

# No ENVIRONMENT for the descriptors: main() applies the compiled-in path, so the binary
# runs standalone and nothing machine-specific reaches the generated CTest files. The
# install-time file is written from this same list, where a build-tree path would name a
# directory that does not exist on the machine running the installed suite.
add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    clang_tidy_check(hip_kernel_provider_census_tests)

    # Every entry for this binary comes from hkp_register_census_tests() above, each naming
    # one shard, one architecture and one filtered suite; a bare run supplies none of those.
    # That is the no-correct-bare-invocation case EXPLICIT_REGISTRATION exists for.
    add_unit_test_target(hip_kernel_provider_census_tests ${CMAKE_CURRENT_BINARY_DIR}
                         EXPLICIT_REGISTRATION)
endif()

# Published one scope up for the same reason the integration tests do it: under YAML
# categorization the raw test name is never registered, so the tiered suites created by
# apply_test_category_labels() in ../CMakeLists.txt are the only place the ambient test
# environment -- an ASAN symbolizer path, a coverage profile destination -- can land.
if(DEFINED hip_kernel_provider_tests_TEST_ENVIRONMENT)
    set(hip_kernel_provider_tests_TEST_ENVIRONMENT
        "${hip_kernel_provider_tests_TEST_ENVIRONMENT}" PARENT_SCOPE)
endif()

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
    core/TestApplicabilityManager.cpp
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
    add_kernels_for_embedding(
        TARGET hip_kernel_provider_tests
        FILES "${_ingestor_kernel_dir}/PointwiseAdd.cpp"
              "${_ingestor_kernel_dir}/PointwiseMul.cpp"
              "${_ingestor_kernel_dir}/PointwiseSub.cpp"
        KEYS kernels/PointwiseAdd.cpp
             kernels/PointwiseMul.cpp
             kernels/PointwiseSub.cpp
    )
    if(HIPKERNELPROVIDER_ENABLE_CONV)
        # The conv engine group's production kernels, under the keys its shipped
        # descriptors name, so the conv tests compile them as the plugin does.
        add_kernels_for_embedding(
            TARGET hip_kernel_provider_tests
            FILES ${HKP_CONV_KERNEL_SOURCES}
            KEYS ${HKP_CONV_KERNEL_KEYS}
        )
    endif()
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
        HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR="${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
        HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}"
        HIPKERNELPROVIDER_UNIT_KPACK_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}/${HIPKERNELPROVIDER_TEST_SET_SHARED}"
        # The packed multi-launch set, which sits beside the unit root instead of under it:
        # the Container never loads it, and the case that reads it builds its own descriptor
        # tree around the archive. Binary-relative for the reason the defines above are.
        HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_MULTI_LAUNCH}")

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
        compilation/TestCodeObjectLoad.cpp
        compilation/TestKpackArchSelection.cpp
        compilation/TestKpackArchive.cpp
        compilation/TestKpackCodeObjectRead.cpp
        compilation/TestKpackGenericArchive.cpp
        compilation/TestKpackKernelLoader.cpp
        compilation/TestKpackModuleCache.cpp
        core/TestPluginSerialization.cpp
        engines/kernel_ingestor_engine/TestIngestorPlanCapture.cpp
        engines/kernel_ingestor_engine/TestIngestorPlanRoundTrip.cpp
        engines/kernel_ingestor_engine/packs/TestPointwiseKpackDispatch.cpp
        ingestor/TestPackedDescriptorLoad.cpp
    )

    # This binary's own packed copy of the archive_fixture Pointwise set. The save cases
    # need a packed Pointwise kernel, and the unit root packs none: its Pointwise set is
    # embedded_source. The copy stages outside the unit root, so the engine does not
    # discover it. Binary-relative, like the defines above.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPKERNELPROVIDER_ARCHIVE_FIXTURE_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_UNIT_ARCHIVE_FIXTURE_SUBDIR}")
    if(TARGET hkp_packaging_unit_archive_fixture)
        add_dependencies(hip_kernel_provider_tests hkp_packaging_unit_archive_fixture)
    endif()

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

    # Install the vendored kpack test fixture beside the test binary so it is
    # available in installed / repackaged layouts (e.g. TheRock CI) where the
    # source tree is absent. The test code falls back to this exe-relative path
    # when the compile-time source path does not exist.
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_zstd.kpack"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/fixtures")

    # This binary reads the per-arch descriptors and archives the packing roots stage under
    # test_arch_content/, and each packing target is reachable only through ALL.
    # Building this binary by name -- what `ninja check` does -- otherwise runs the cases
    # against a directory that does not exist yet, and they skip rather than fail.
    # Each descriptor set has its own staging target. Name each one individually.
    foreach(_hkp_set hkp_packaging_unit hkp_packaging_unit_shared)
        if(TARGET ${_hkp_set})
            add_dependencies(hip_kernel_provider_tests ${_hkp_set})
            add_dependencies(hip_kernel_provider_census_tests ${_hkp_set})
        endif()
    endforeach()

    # The multi-launch set is read by this binary alone, and by no census entry.
    if(TARGET hkp_packaging_multi_launch)
        add_dependencies(hip_kernel_provider_tests hkp_packaging_multi_launch)
    endif()

    # FlyDSL's kernels ship in the production shard, which the unit root main()
    # resolves does not reach, so FlyDSL's device cases would find no descriptor for
    # their engine. The binary therefore names the production root as an ADDITIVE one,
    # as the integration binary does, and is built after the pack that fills it.
    #
    # The ordinary binary only. A census addresses exactly one shard through an explicit
    # HIPDNN_DESCRIPTOR_DIR, and a second root reachable from that process would put
    # kernels in its inventory that the entry running it never named.
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
        if(TARGET hkp_packaging_product)
            add_dependencies(hip_kernel_provider_tests hkp_packaging_product)
        endif()
        target_compile_definitions(hip_kernel_provider_tests PRIVATE
            HIPKERNELPROVIDER_FLYDSL_PRODUCT_DESCRIPTOR_RELDIR="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}")
    endif()

    # The packed-load test also reads the slotted fixtures' shards (RFC 0021), beside the
    # unit tree. The fixed-arch roots pack only for their arch (_hkp_wire_fixed_arch_root);
    # the test decides from the staged shards, so no define follows GPU_TARGETS. rocKE is
    # the only thing it is told.
    foreach(_hkp_slotted hkp_packaging_slotted hkp_packaging_slotted_hsaco
                         hkp_packaging_slotted_multi_launch)
        if(TARGET ${_hkp_slotted})
            add_dependencies(hip_kernel_provider_tests ${_hkp_slotted})
        endif()
    endforeach()
    if(HIPKERNELPROVIDER_ENABLE_ROCKE)
        target_compile_definitions(hip_kernel_provider_tests PRIVATE
            HIPKERNELPROVIDER_ENABLE_ROCKE)
    endif()

    # The census binary additionally reads the production shard, which is wired only
    # where a production root holds descriptors this build packs for. Absent, there is
    # nothing to depend on and no entry is registered against it either.
    if(TARGET hkp_packaging_product)
        add_dependencies(hip_kernel_provider_census_tests hkp_packaging_product)
    endif()

    # The parent of the unit descriptor sets, and the only root this binary reads.
    # A walk of a parent reaches every subtree staged under it, so a further pack
    # there comes under the check with no edit here.
    hkp_verify_embedded_sources(
        TARGET hip_kernel_provider_tests
        STAGED_DESCRIPTOR_ROOTS ${HIPKERNELPROVIDER_UNIT_BUILD_DIR}
        PACK_NAMES unit unit_shared
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

    # FlyDSL's kernels ship in the production shard, so their census addresses that
    # pack, narrowed to the architectures FlyDSL ships for: the product pack may be
    # wired for an architecture FlyDSL has no objects for, where these suites would find
    # no set. ACTIVE rather than the option: with the option ON for an architecture no
    # kernels are checked in for, FlyDSL stands down and there is nothing to census.
    #
    # The TestFlydsl*Packs suites are the FlyDSL suites that qualify, and they have files
    # of their own in the census binary for that reason. FlyDSL's GPU suites skip, which a
    # census entry treats as a failure; its matcher-acceptance suites are TEST_P; and its
    # remaining host suites build graphs in-process instead of reading loadedSet(), so
    # censusing one would stay green against an empty shard -- the exact miss the census
    # exists to catch. Every case below opens by reading the staged set.
    #
    # Each op's census reads only the shards that op ships objects for
    # (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>): ops ship for different architectures, and a
    # suite addressing a shard its op is absent from would fail.
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_rmsnorm)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_rmsnorm}
            SUITES TestFlydslRmsNormPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsTheGenericTierPerFormAndBakedTiersFor16Bit
                UnderstandsRuntimeEpsilonGraphsButNotAlignedOnes
                EveryKernelDeclaresTheNineSlotSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_rmsnorm_bwd)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_rmsnorm_bwd}
            SUITES TestFlydslRmsNormBwdPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsTheGenericTiersPerFormAndBakedWidthsFor16Bit
                EveryKernelDeclaresThePartialKernelsSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna}
            SUITES TestFlydslSdpaCdnaPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsUniqueObjectsAndOneTopServeAllPerClass
                RanksTheNarrowerBucketFirst
                EveryKernelDeclaresItsFamilysSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna_dualwave)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna_dualwave}
            SUITES TestFlydslSdpaCdnaDualwavePacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsUniqueObjectsAndOneTopServeAllPerClass
                EveryKernelDeclaresItsFamilysSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna_decode)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna_decode}
            SUITES TestFlydslSdpaCdnaDecodePacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsUniqueObjectsAndOneTopServeAllPerClass
                EveryKernelDeclaresItsFamilysSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_rdna)
        # A second suite over the same shard: one EXPECTED_CASES list pins one suite.
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_rdna}
            SUITES TestFlydslSdpaRdnaPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsEverySpecializedAndGenericVariantOnce
                RanksTheNarrowestServingObjectFirst
                EveryKernelDeclaresTheThirtyEightSlotSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_aiw)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_aiw}
            SUITES TestFlydslSdpaAiwPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsEveryVariantOnce
                EveryKernelDeclaresTheFiftySlotSignature
        )
    endif()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_rdna_decode)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_rdna_decode}
            SUITES TestFlydslSdpaRdnaDecodePacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsEveryDecodeVariantOnce
                RanksTheNarrowestServingObjectFirst
                EveryKernelDeclaresTheFortySlotSignature
        )
    endif()

    # Census for the gfx950 attention production packs. Each reads only the product
    # shard, so it is eligible for census registration.
    #
    # The predicate and ARCHES narrow independently -- whether `product` carries THIS
    # bundle at all, and which of its wired architectures the bundle emits for. Either
    # alone lets a census address a shard its suite has nothing to say about.
    #
    # The Pointwise call above needs neither: its fixtures are the unit root itself, and
    # they cover every architecture it is wired for.
    #
    # Each item is "<snake slug>|<engine local name>|<hand-written companion cases>": the
    # generated Test<local>Packs suite pins the same cases for every engine, and the
    # companion Test<local>CatalogTiles suite checks every shipped kernel against that
    # engine's own launchability rule. The companion gets its own call because a pin names
    # one suite's cases. _hkp_gfx950_attention_<snake> is left set for the cross-engine
    # suite below.
    foreach(_gfx950_engine IN ITEMS
            "dense|Gfx950AttentionDense|EveryShippedKernelsTileIsOneTheEngineLaunches"
            "unified_2d|Gfx950AttentionUnified2D|EveryShippedKernelsGeometryIsOneTheEngineLaunches"
            "unified_3d|Gfx950AttentionUnified3D|EveryShippedKernelsGeometryIsOneTheEngineLaunches,EveryEntryLaunchesItsSegmentAndReduceKernels")
        string(REPLACE "|" ";" _gfx950_engine "${_gfx950_engine}")
        list(GET _gfx950_engine 0 _snake)
        list(GET _gfx950_engine 1 _local)
        list(GET _gfx950_engine 2 _companion_cases)
        string(REPLACE "," ";" _companion_cases "${_companion_cases}")

        hkp_product_ships_engine_for_arch(_hkp_gfx950_attention_${_snake}
                                          "hipkernel:${_local}" gfx950)
        if(_hkp_gfx950_attention_${_snake})
            hkp_register_census_tests(
                TARGET hip_kernel_provider_census_tests
                PACK_NAME product
                ARCHES gfx950
                SUITES Test${_local}Packs
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
            hkp_register_census_tests(
                TARGET hip_kernel_provider_census_tests
                PACK_NAME product
                ARCHES gfx950
                SUITES Test${_local}CatalogTiles
                EXPECTED_CASES ${_companion_cases}
            )
        else()
            message(STATUS
                "hkp: this configuration does not ship hipkernel:${_local} for "
                "gfx950, so its census is not registered.")
        endif()
    endforeach()

    # Unified2D hands decode to Unified3D by a static rule, so with both engines shipped,
    # every 2D cell that rule hands off must have a 3D kernel.
    if(_hkp_gfx950_attention_unified_2d AND _hkp_gfx950_attention_unified_3d)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx950
            SUITES TestGfx950AttentionUnified2DDecodeHandoff
            EXPECTED_CASES
                EveryDecodeCellHandedToUnified3DHasAUnified3DKernel
        )
    endif()

    # Census for the gfx942 unified 3D (split-KV decode) production pack; gated like
    # gfx950's above.
    hkp_gfx942_attention_unified_3d_available(_hkp_gfx942_attention_unified_3d)
    if(_hkp_gfx942_attention_unified_3d)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx942
            SUITES TestGfx942AttentionUnified3dPacks
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
            "hkp: this configuration does not ship hipkernel:Gfx942AttentionUnified3d for "
            "gfx942, so its census is not registered.")
    endif()

    # One generic kernel set per RDNA family ships in every RDNA3/3.5 and RDNA4 shard,
    # as two engines: the prefill objects and the split-KV decode objects (the wide-Q
    # prefill and D256 decode geometries differ per family). The census checks each wired
    # shard against its own inventory. Gated per arch like gfx950 above.
    foreach(_engine_suite IN ITEMS
            "hipkernel:Gfx1151WmmaAttention|TestGfx1151WmmaAttentionPacks"
            "hipkernel:Gfx1151WmmaAttentionDecode|TestGfx1151WmmaAttentionDecodePacks")
        string(REPLACE "|" ";" _engine_suite "${_engine_suite}")
        list(GET _engine_suite 0 _hkp_engine)
        list(GET _engine_suite 1 _hkp_suite)
        set(_hkp_rdna_wmma_attention_arches "")
        foreach(_arch IN ITEMS gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153
                               gfx1200 gfx1201)
            hkp_product_ships_engine_for_arch(_hkp_ships "${_hkp_engine}" ${_arch})
            if(_hkp_ships)
                list(APPEND _hkp_rdna_wmma_attention_arches ${_arch})
            endif()
        endforeach()
        if(_hkp_rdna_wmma_attention_arches)
            hkp_register_census_tests(
                TARGET hip_kernel_provider_census_tests
                PACK_NAME product
                ARCHES ${_hkp_rdna_wmma_attention_arches}
                SUITES ${_hkp_suite}
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
                "hkp: this configuration does not ship ${_hkp_engine} for any RDNA arch, "
                "so its census is not registered.")
        endif()
    endforeach()
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

# The conv RDNA generic-target compile check (conv/tools/rdna_generic/check_generic.py): a
# sample of logged conv configs, each compiled for its RDNA chip and for the chip's LLVM
# generic target with the ROCm clang. It fails when a config does not compile or when its
# generic build loses the chip build's matrix-core instructions (a WMMA config fallen back to
# its trap). Host-only and minutes long, so opt-in and labeled out of the tiered suites.
if(HIPKERNELPROVIDER_CONV_GENERIC_COMPILE_TEST AND HIPDNN_ENABLE_KERNEL_INGESTOR
   AND HIPKERNELPROVIDER_ENABLE_CONV)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(DEFINED ROCM_PATH)
        set(_hkp_generic_rocm "${ROCM_PATH}")
    else()
        set(_hkp_generic_rocm "/opt/rocm")
    endif()
    set(_hkp_generic_tools "${CMAKE_CURRENT_SOURCE_DIR}/../../conv/tools/rdna_generic")
    enable_testing()
    add_test(
        NAME hkp_conv_rdna_generic_compile
        COMMAND
            ${Python3_EXECUTABLE} ${_hkp_generic_tools}/check_generic.py
            --configs ${_hkp_generic_tools}/rdna_generic_sample.jsonl
            --kernels-root ${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/descriptors/conv
            --rocm ${_hkp_generic_rocm} -j 2)
    set_tests_properties(hkp_conv_rdna_generic_compile PROPERTIES
        LABELS "conv_generic_compile" TIMEOUT 3600)
endif()

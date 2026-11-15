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

    # The production descriptor tree, as the same binary-relative offset the integration
    # binary uses. Only TestGfx950ConvDgradMatchers reads it, to run the shipped dgrad
    # catalog's metadata through the real state manager; it skips when this configuration
    # packs no production tree, so no dependency on hkp_packaging_product is added here.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR="${HIPKERNELPROVIDER_PLUGIN_ENGINE_RELDIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}")

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
                ShipsTheF32Objects
                RanksTheNarrowerBucketFirst
                ShipsTheWideAndPairBuckets
                StagesTheHeadDim256KeysInChunksOn64KiBParts
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
                ServesHeadDim256OnFourWaves
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
                EveryKernelDeclaresTheFortySevenSlotSignature
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
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_bwd_merged)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_bwd_merged}
            SUITES TestFlydslSdpaBwdPacks
            EXPECTED_CASES
                IsOnePackWithAGraphMatchAndOneKernelScopedMatcher
                ShipsEveryMergedRowOnce
                EveryKernelDeclaresTheMergedSignature
                DeclaresASchemaAtLeastTheRaggedOffsetMultiplierFloor
                ServesStampedRaggedGraphsAndDeclinesTheFieldsTheMatcherRejects
                RefusesPlanSerialization
        )
    endif()
    # The split engine's slots are ops of their own (prep, dkdv, dq); its descriptor-only
    # engine folder has none. The census reads a shard only for an arch where all three slots
    # ship, and the suite keys its inventory on the arch the entry names: gfx950 content for
    # gfx950, gfx11-generic content for every gfx11 member. The merged engine's census above
    # is registered only where merged content ships (gfx950).
    set(_hkp_split_arches)
    foreach(_arch IN LISTS HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_bwd_dq)
        if(_arch IN_LIST HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_bwd_prep
           AND _arch IN_LIST HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_bwd_dkdv)
            list(APPEND _hkp_split_arches ${_arch})
        endif()
    endforeach()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND _hkp_split_arches)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${_hkp_split_arches}
            SUITES TestFlydslSdpaBwdSplitPacks
            EXPECTED_CASES
                IsOneSlottedEngineWithANativeGraphMatchAndAKernelMatcherPerSlot
                ShipsEverySplitRowOnceAndNoKernelInTwoSlots
                EveryKernelDeclaresItsSlotsSignature
                EveryVerdictRowIsServedOrDeclinedAsTheTableSays
                EveryServedRowsSlotCatalogsHaveUniqueEnrolledKnobTuples
                DeclaresASchemaAtLeastTheRaggedOffsetMultiplierFloor
                ServesStampedRaggedGraphsAndDeclinesTheFieldsTheMatcherRejects
                RefusesPlanSerialization
                TheChunkContractKeepsAnAlignedBiasOffDirectRowsWithAStagedAlternative
                EveryProbeGraphIsServedByTheRowsTheModelNames
                TheCoverageConstantsAreTheHeadDimsTheShippedParityRowsServe
        )
    endif()
    # The host half of the split engine's parity cases checks the gfx11 shard against the
    # matcher contract. The suite is compiled into the census binary whatever GPU_TARGETS
    # holds: TheRock installs one copy of that binary for every family's test job, so a
    # copy built without gfx11 targets must still contain the suite a gfx11 entry runs. The
    # suite skips, which a census entry reports as a failure, when the shard it is handed
    # holds no gfx11 pack. Only the registration is per arch: an entry for each gfx11 arch
    # the three slot ops ship for, and none in a configuration without one.
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
        target_sources(hip_kernel_provider_census_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/engines/kernel_ingestor_engine/packs/TestFlydslSdpaBwdParityGraphs.cpp
        )
    endif()
    set(_hkp_gfx11_split_arches)
    foreach(_arch IN LISTS _hkp_split_arches)
        if(_arch MATCHES "^gfx11")
            list(APPEND _hkp_gfx11_split_arches ${_arch})
        endif()
    endforeach()
    if(HIPKERNELPROVIDER_FLYDSL_ACTIVE AND _hkp_gfx11_split_arches)
        # The pin names the parameterised cases of FlydslSdpaBwdParityCases.hpp, one per
        # case of the shared list the device half also launches.
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES ${_hkp_gfx11_split_arches}
            SUITES TestFlydslSdpaBwdParityGraphs
            EXPECTED_CASES
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesBottomRightCausalCrossAttentionWithFewerQueriesThanKeys
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesBottomRightCausalCrossAttentionWithMoreQueriesThanKeys
                IsServedOrDeclinedAsTheMatcherContractStates/AppliesASlidingWindowNarrowerThanATile
                IsServedOrDeclinedAsTheMatcherContractStates/AppliesALeftOnlyWindow
                IsServedOrDeclinedAsTheMatcherContractStates/AppliesABandToTheRightOfTheDiagonal
                IsServedOrDeclinedAsTheMatcherContractStates/AppliesAWideBandWithMoreQueriesThanKeys
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAValueHeadDimNarrowerThanTheQueryOne
                IsServedOrDeclinedAsTheMatcherContractStates/ServesANarrowerValueHeadDimOverGroupedCrossAttention
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAValueHeadDimWiderThanTheQueryOne
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAWiderValueHeadDimWithABias
                IsServedOrDeclinedAsTheMatcherContractStates/ZeroesRowsNoKeyReaches
                IsServedOrDeclinedAsTheMatcherContractStates/ZeroesRowsATopLeftCausalWindowStartsPastTheKeys
                IsServedOrDeclinedAsTheMatcherContractStates/ZeroesRowsALeftOnlyWindowStartsPastTheKeys
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInBf16
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInF16
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInFp32
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsWithMinusInfinityInBf16
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsWithMinusInfinityInF16
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksWholeRowsWithMinusInfinityInFp32
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksLeadingKeysWithMinusInfinityInBf16
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasThatMasksLeadingKeysWithMinusInfinityInFp32
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesHeadDim256CausalGroupedQueryAttention
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesHeadDim256NonCausal
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesAGenericHeadDim40
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesAGenericHeadDim80
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesAGenericHeadDim152
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesAGenericHeadDim192
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesAGenericHeadDim240
                IsServedOrDeclinedAsTheMatcherContractStates/AddsAPerHeadBiasAtHeadDim128
                IsServedOrDeclinedAsTheMatcherContractStates/AddsABiasInTheQueryDtype
                IsServedOrDeclinedAsTheMatcherContractStates/AddsAnFp32BiasToAnFp32Query
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAnFp32BiasBesideAHalfQuery
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAnFp32BiasBesideABf16Query
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAnFp32BiasAndDbiasBesideAHalfQuery
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesADbiasWhoseRowsAreNotAWholeVectorApart
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesABiasPitchThatLeavesTheDbiasUnaligned
                IsServedOrDeclinedAsTheMatcherContractStates/ReadsKAndVThroughTheirHeadGroups
                IsServedOrDeclinedAsTheMatcherContractStates/ReadsKAndVThroughOneHeadForAMultiQueryGroup
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAVWithHeadsOfItsOwn
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAVWithMoreHeadsThanKUnderGroupedQueries
                IsServedOrDeclinedAsTheMatcherContractStates/ServesAVWithFewerHeadsThanKUnderCausalGroupedQueries
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesVHeadsOfItsOwnWithABias
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesVHeadsOfItsOwnUnderFp32
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesVHeadsOfItsOwnWithRaggedKeys
                IsServedOrDeclinedAsTheMatcherContractStates/DeclinesQueryHeadsThatAreNotAMultipleOfTheValueHeads
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesHeadMajorStrides
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesSequenceMajorStrides
                IsServedOrDeclinedAsTheMatcherContractStates/ComputesPackedQkvViews
                IsServedOrDeclinedAsTheMatcherContractStates/ReadsVAndDoAndTheLseThroughTheirOwnLayouts
                IsServedOrDeclinedAsTheMatcherContractStates/ReadsARuntimeScaleAtExecute
                IsServedOrDeclinedAsTheMatcherContractStates/AppliesTheDefaultScaleWhenTheGraphGivesNone
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
                EveryKernelDeclaresTheFortyFourSlotSignature
        )
    endif()

    # The cases every generated Test<Engine>Packs suite below registers, shared by the
    # attention and RDNA calls.
    set(_hkp_generated_packs_cases
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

    # A suite whose engine config declares behavior_notes renders one more case. The list
    # names those suites; keep it in step with the configs' engine.behavior_notes.
    set(_hkp_behavior_notes_suites
        TestGfx950AttentionDensePacks
        TestGfx1151WmmaAttentionPacks
        TestGfx1151WmmaAttentionDecodePacks
    )
    # _hkp_generated_packs_cases_for(<out> <suite>): the cases <suite> registers.
    function(_hkp_generated_packs_cases_for out_var suite)
        set(_cases ${_hkp_generated_packs_cases})
        if(suite IN_LIST _hkp_behavior_notes_suites)
            list(APPEND _cases DeclaresItsConfiguredBehaviorNotes)
        endif()
        set(${out_var} ${_cases} PARENT_SCOPE)
    endfunction()

    # Census for each HKP_PINNED_ATTENTION_ENGINES engine's production pack (the table is in
    # ../CMakeLists.txt). Each reads only the product shard, so it is eligible for census
    # registration.
    #
    # The predicate and ARCHES narrow independently -- whether `product` carries THIS
    # bundle at all, and which of its wired architectures the bundle emits for. Either
    # alone lets a census address a shard its suite has nothing to say about.
    #
    # The Pointwise call above needs neither: its fixtures are the unit root itself, and
    # they cover every architecture it is wired for.
    #
    # An engine with hand-written companion suites lists each one's cases below under
    # _hkp_<stem>_cases_<arch>_<snake slug>, the companion's suite being
    # Test<local><Suite> for each <Suite> <stem> pair in _hkp_companion_suites. Each gets its
    # own call because a pin names one suite's cases. _hkp_<arch>_attention_<snake slug> is
    # left set for the cross-engine suite below.
    #   CatalogTiles checks every shipped kernel against the engine's own launchability rule.
    #   ModelBinding fails the build when a shipped trained ranker would be refused at
    #   runtime, which the engine otherwise only logs before ranking on priority. A model
    #   whose model.bin was not pulled is packed without its UHD, with a warning and an entry
    #   in hkp-unpulled-models.record; the suite tolerates an absent UHD only when that record
    #   names it, and never when HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS is ON, which it
    #   learns from the compile definition below.
    set(_hkp_companion_suites CatalogTiles catalog_tiles ModelBinding model_binding)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/kernel_ingestor_engine/packs/TestGfx942AttentionUnified3dModelBinding.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/engines/kernel_ingestor_engine/packs/TestGfx1250ConvDgradAotModelBinding.cpp"
        PROPERTIES COMPILE_DEFINITIONS
            "HKP_REQUIRE_DVC_DESCRIPTORS=$<BOOL:${HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS}>")
    set(_hkp_catalog_tiles_cases_gfx950_dense EveryShippedKernelsTileIsOneTheEngineLaunches)
    set(_hkp_catalog_tiles_cases_gfx950_unified_2d
        EveryShippedKernelsGeometryIsOneTheEngineLaunches)
    set(_hkp_catalog_tiles_cases_gfx950_unified_3d
        EveryShippedKernelsGeometryIsOneTheEngineLaunches
        EveryEntryLaunchesItsSegmentAndReduceKernels)
    set(_hkp_catalog_tiles_cases_gfx942_dense EveryShippedKernelsTileIsOneTheEngineLaunches)
    set(_hkp_model_binding_cases_gfx942_unified_3d
        TheBoundL2KernelRankerLoads
        TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)

    set(_engines ${HKP_PINNED_ATTENTION_ENGINES})
    while(_engines)
        list(POP_FRONT _engines _arch _snake _local _filter_var _toml _categories_var)
        hkp_product_ships_engine_for_arch(_hkp_${_arch}_attention_${_snake}
                                          "hipkernel:${_local}" ${_arch})
        if(_hkp_${_arch}_attention_${_snake})
            _hkp_generated_packs_cases_for(_hkp_cases Test${_local}Packs)
            hkp_register_census_tests(
                TARGET hip_kernel_provider_census_tests
                PACK_NAME product
                ARCHES ${_arch}
                SUITES Test${_local}Packs
                EXPECTED_CASES ${_hkp_cases}
            )
        else()
            message(STATUS
                "hkp: this configuration does not ship hipkernel:${_local} for "
                "${_arch}, so its census is not registered.")
        endif()
        set(_companions ${_hkp_companion_suites})
        while(_companions)
            list(POP_FRONT _companions _suite _stem)
            if(_hkp_${_arch}_attention_${_snake}
               AND DEFINED _hkp_${_stem}_cases_${_arch}_${_snake})
                hkp_register_census_tests(
                    TARGET hip_kernel_provider_census_tests
                    PACK_NAME product
                    ARCHES ${_arch}
                    SUITES Test${_local}${_suite}
                    EXPECTED_CASES ${_hkp_${_stem}_cases_${_arch}_${_snake}}
                )
            endif()
            unset(_hkp_${_stem}_cases_${_arch}_${_snake})
        endwhile()
    endwhile()
    # The loop unsets each row's entries, so one left set names no row and its census would
    # silently never register.
    set(_hkp_companion_stems "")
    set(_companions ${_hkp_companion_suites})
    while(_companions)
        list(POP_FRONT _companions _suite _stem)
        list(APPEND _hkp_companion_stems ${_stem})
    endwhile()
    list(JOIN _hkp_companion_stems "|" _hkp_companion_stems)
    get_directory_property(_hkp_unmatched VARIABLES)
    list(FILTER _hkp_unmatched INCLUDE REGEX "^_hkp_(${_hkp_companion_stems})_cases_")
    if(_hkp_unmatched)
        message(FATAL_ERROR
            "hkp: ${_hkp_unmatched} names no HKP_PINNED_ATTENTION_ENGINES row's <arch> and "
            "<snake slug>, so its companion census would never register.")
    endif()

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

    # One generic kernel set per RDNA family ships in every RDNA3/3.5 and RDNA4 shard,
    # as two engines: the prefill objects and the split-KV decode objects (the wide-Q
    # prefill and D256 decode geometries differ per family). The census checks each wired
    # shard against its own inventory. Gated per arch like the attention engines above.
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
            _hkp_generated_packs_cases_for(_hkp_cases ${_hkp_suite})
            hkp_register_census_tests(
                TARGET hip_kernel_provider_census_tests
                PACK_NAME product
                ARCHES ${_hkp_rdna_wmma_attention_arches}
                SUITES ${_hkp_suite}
                EXPECTED_CASES ${_hkp_cases}
            )
            # The prefill engine's page sizes and architectures, stated once in
            # Gfx1151WmmaAttentionCatalog.hpp for the matcher and the GPU suites, are
            # checked against the loaded objects. The suite is registered at every wired
            # arch the header lists, read from the header itself, not only where the
            # engine ships: a listed arch the catalog drops then loads no set and fails, as
            # an object shipped outside the list does.
            if(_hkp_engine STREQUAL "hipkernel:Gfx1151WmmaAttention")
                set(_hkp_catalog_hpp
                    "${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/packs/Gfx1151WmmaAttentionCatalog.hpp")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_hkp_catalog_hpp}")
                file(READ "${_hkp_catalog_hpp}" _hkp_stated_arches)
                string(REGEX MATCH "GFX1151_WMMA_ATTENTION_ARCHS\\{[^}]*\\}" _hkp_stated_arches
                       "${_hkp_stated_arches}")
                string(REGEX MATCHALL "gfx[0-9a-f]+" _hkp_stated_arches "${_hkp_stated_arches}")
                if(NOT _hkp_stated_arches)
                    message(FATAL_ERROR
                        "hkp: no GFX1151_WMMA_ATTENTION_ARCHS list found in ${_hkp_catalog_hpp}")
                endif()
                hkp_register_census_tests(
                    TARGET hip_kernel_provider_census_tests
                    PACK_NAME product
                    ARCHES ${_hkp_stated_arches}
                    SUITES TestGfx1151WmmaAttentionCatalog
                    EXPECTED_CASES
                        ShipsPagedObjectsForExactlyTheMatchersPageSizes
                        ShipsForNoArchitectureOutsideTheStatedList
                )
            endif()
        else()
            message(STATUS
                "hkp: this configuration does not ship ${_hkp_engine} for any RDNA arch, "
                "so its census is not registered.")
        endif()
    endforeach()

    # Census for the shape-generic backward-data AOT-instance pack,
    # hipkernel:Gfx950ConvDgradAot.
    hkp_product_ships_engine_for_arch(_hkp_gfx950_conv_dgrad_aot
                                      "hipkernel:Gfx950ConvDgradAot" gfx950)
    if(_hkp_gfx950_conv_dgrad_aot)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx950
            SUITES TestGfx950ConvDgradAotPacks
            EXPECTED_CASES ${_hkp_generated_packs_cases}
        )
    else()
        message(STATUS
            "hkp: this configuration does not ship hipkernel:Gfx950ConvDgradAot for "
            "gfx950, so its census is not registered.")
    endif()

    # ... and its gfx1250 sibling, hipkernel:Gfx1250ConvDgradAot.
    hkp_product_ships_engine_for_arch(_hkp_gfx1250_conv_dgrad_aot
                                      "hipkernel:Gfx1250ConvDgradAot" gfx1250)
    if(_hkp_gfx1250_conv_dgrad_aot)
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx1250
            SUITES TestGfx1250ConvDgradAotPacks
            EXPECTED_CASES ${_hkp_generated_packs_cases}
        )
        # Its trained ranker, as the ModelBinding companion above checks the attention
        # engine's: the bound model loads and scores every shipped kernel.
        hkp_register_census_tests(
            TARGET hip_kernel_provider_census_tests
            PACK_NAME product
            ARCHES gfx1250
            SUITES TestGfx1250ConvDgradAotModelBinding
            EXPECTED_CASES
                TheBoundL2KernelRankerLoads
                TheBoundL2KernelRankerScoresEveryShippedKernel
        )
    else()
        message(STATUS
            "hkp: this configuration does not ship hipkernel:Gfx1250ConvDgradAot for "
            "gfx1250, so its census is not registered.")
    endif()

endif()

clang_tidy_check(hip_kernel_provider_tests)

# No ENVIRONMENT for the descriptors: main() applies the compiled-in path, so the binary
# runs standalone and nothing machine-specific reaches the generated CTest files. The
# install-time file is written from this same list, where a build-tree path would name a
# directory that does not exist on the machine running the installed suite.
add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})

# The FlyDSL backward suites that build graphs and match rows in-process, with no device
# (the device cases are the TestGpuFlydslSdpaBwd* suites, which the host label excludes).
# The tier suites above run them too, but only the host label reaches a runner without a
# GPU, so they get an entry of their own. Registered beside the drift-guard entry
# (flydsl/CMakeLists.txt), with the same labels and the same YAML tier entry.
if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
    add_test(NAME hip-kernel-provider-flydsl-sdpa-bwd-host
             COMMAND hip_kernel_provider_tests "--gtest_filter=TestFlydslSdpaBwd*")
    set_tests_properties(hip-kernel-provider-flydsl-sdpa-bwd-host PROPERTIES
        LABELS "unit_test;hip-kernel-provider;host"
        ENVIRONMENT "${TEST_ENVIRONMENT}"
        TIMEOUT 600)
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endif()

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

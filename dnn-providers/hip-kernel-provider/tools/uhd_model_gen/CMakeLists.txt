# Build-time producer of the pointwise_model pack's binary artifacts.
#
# Only useful when the ingestor is on -- it is the only thing that reads a UHD -- and the
# feature extractor it links is gated the same way.
if(NOT HIPDNN_ENABLE_KERNEL_INGESTOR)
    return()
endif()

# A host tool, not part of the provider library: it runs during the build to produce data,
# and nothing links it.
add_executable(uhd_model_gen UhdModelGen.cpp)

# The flatbuffers SDK for the two schemas, the plugin SDK for FeatureExtractor::computeHash
# -- the same function the runtime validates the emitted hash against. Not the test SDK,
# whose GbdtModelTestBuilder overlaps: a shipped tool must not reach into test scaffolding.
target_link_libraries(uhd_model_gen PRIVATE hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

target_compile_options(uhd_model_gen PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# Build-time generator of the pointwise_model pack's UHD and model artifact.
if(NOT HIPDNN_ENABLE_KERNEL_INGESTOR)
    return()
endif()

# Host tool run during the build; nothing links it.
add_executable(uhd_model_gen UhdModelGen.cpp)

# Not the test SDK: a shipped tool must not depend on test scaffolding.
target_link_libraries(uhd_model_gen PRIVATE hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hip::host)
target_compile_definitions(uhd_model_gen PRIVATE __HIP_PLATFORM_AMD__)

target_compile_options(uhd_model_gen PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

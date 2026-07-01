target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationGpuPointwiseReluForward.cpp
        IntegrationGpuPointwiseTanhForward.cpp
        IntegrationGpuPointwiseSigmoidForward.cpp
)

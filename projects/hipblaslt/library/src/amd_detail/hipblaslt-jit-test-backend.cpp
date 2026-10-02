// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-mock.hpp"
#include "rocblaslt_secure_env.hpp"

// The provider of HIPBLASLT_JIT_TESTING builds: heuristic queries replay the
// bundle HIPBLASLT_JIT_TEST_REPLAY names through the mock backend.
namespace hipblaslt_jit
{
    Status makeDefaultProcessBackend(ProcessBackend& made)
    {
        const char* replay = rocblaslt_secure_getenv("HIPBLASLT_JIT_TEST_REPLAY");
        if(!replay || !*replay)
            return {Status::Code::Failed,
                    Stage::Configure,
                    "The JIT test backend has no bundle to replay; set HIPBLASLT_JIT_TEST_REPLAY"};
        hipblaslt_ext::experimental::jit::mock::Options options;
        options.replay = replay;
        made.backend   = hipblaslt_ext::experimental::jit::mock::makeBackend(options);
        return {};
    }
}

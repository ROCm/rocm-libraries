#pragma once

#include <cstdint>

namespace hipdnn_tests::plugin_constants
{
template <class T>
constexpr int64_t engineId() = delete;
} // namespace hipdnn_tests::plugin_constants

#define HIPDNN_MAP_TO_ID(ClassName, id)      \
    class ClassName;                         \
    namespace hipdnn_tests::plugin_constants \
    {                                        \
    template <>                              \
    constexpr int64_t engineId<ClassName>()  \
    {                                        \
        return id;                           \
    };                                       \
    }

HIPDNN_MAP_TO_ID(GoodPlugin, -2);
HIPDNN_MAP_TO_ID(GoodDefaultPlugin, -3);
HIPDNN_MAP_TO_ID(NoApplicableEnginesAPlugin, -4);
HIPDNN_MAP_TO_ID(NoApplicableEnginesBPlugin, -5);
HIPDNN_MAP_TO_ID(ExecuteFailsPlugin, -6);
HIPDNN_MAP_TO_ID(DuplicateIdAPlugin, -7);
HIPDNN_MAP_TO_ID(DuplicateIdBPlugin, -7);

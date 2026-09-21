// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <test_plugins/TestPluginCommon.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <filesystem>
#include <string>
#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace
{

class ScopedTempFile
{
public:
    explicit ScopedTempFile(std::filesystem::path path)
        : _path(std::move(path))
    {
        std::filesystem::remove(_path);
    }

    ~ScopedTempFile()
    {
        std::error_code ignored;
        std::filesystem::remove(_path, ignored);
    }

    ScopedTempFile(const ScopedTempFile&) = delete;
    ScopedTempFile& operator=(const ScopedTempFile&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

    std::string string() const
    {
        return _path.string();
    }

private:
    std::filesystem::path _path;
};

int currentProcessId()
{
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}

} // namespace

TEST(IntegrationPluginInfrastructure, NoLoadMissThenScopedOwnerHitAndUnload)
{
    const auto sourcePath = test_plugin_internal::resolvePluginPathRelativeToBackend(
        hipdnn_tests::plugin_constants::testSecondOverridePluginPath());
    const auto probePath
        = std::filesystem::temp_directory_path()
          / ("hipdnn_second_override_noload_probe_" + std::to_string(currentProcessId())
             + sourcePath.extension().string());
    const ScopedTempFile probeFile(probePath);
    std::filesystem::copy_file(
        sourcePath, probeFile.path(), std::filesystem::copy_options::overwrite_existing);

    const auto pluginPath = probeFile.string();
    constexpr const char* SUFFIX = "SecondOverride";

    ASSERT_EQ(getLastCallRecordIfLoaded(pluginPath, SUFFIX), nullptr)
        << "RTLD_NOLOAD lookup must not load a plugin on miss.";

    {
        const ScopedTestPluginLibrary owner(pluginPath);
        ASSERT_NE(getLastCallRecordIfLoaded(pluginPath, SUFFIX), nullptr)
            << "RTLD_NOLOAD lookup should resolve while a real owner pins the plugin.";

        resetLastCallRecord(owner, SUFFIX);
        const auto* record = getLastCallRecord(owner, SUFFIX);
        ASSERT_NE(record, nullptr);
        EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
    }

    EXPECT_EQ(getLastCallRecordIfLoaded(pluginPath, SUFFIX), nullptr)
        << "The temporary RTLD_NOLOAD handle must not keep the plugin loaded after owner teardown.";
}

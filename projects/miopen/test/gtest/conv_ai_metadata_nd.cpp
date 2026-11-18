// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/db_path.hpp>
#include <miopen/errors.hpp>
#include <miopen/filesystem.hpp>

#include <gtest/gtest.h>
#include "gtest_common.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

MIOPEN_LIB_ENV_VAR(MIOPEN_SYSTEM_DB_PATH)

namespace {

// A metadata file whose `derived_features` key is present but not an array of strings must fail the
// load with a named cause (MetadataND::LoadDerivedFeatures throws) instead of silently producing an
// empty list and a later engineered-width mismatch. Redirects MIOPEN_SYSTEM_DB_PATH to a temp dir
// holding a copy of a real metadata with that one field corrupted, so every other field still loads
// and the ctor reaches the derived_features parse. CPU only (file load + parse; no device). This
// file is kept separate from conv_ai_nd_heuristics_test.cpp because gtest_common.hpp and
// <fdeep/fdeep.hpp> define conflicting `half` type aliases.
TEST(CPU_ConvAiMetadataND_NONE, ThrowsOnMalformedDerivedFeatures)
{
    namespace fs   = miopen::fs;
    const auto ref = miopen::GetSystemDbPath() / "gfx942_metadata.tn.model";
    if(!fs::exists(ref))
        GTEST_SKIP() << "reference metadata not installed at " << ref;

    nlohmann::json meta;
    {
        std::ifstream in(ref.string());
        in >> meta;
    }
    meta["derived_features"] = 42; // present, but not a list of strings

    const auto dir = fs::temp_directory_path() / "miopen_md_malformed_derived_features";
    fs::create_directories(dir);
    {
        std::ofstream out((dir / "gfx942_metadata.tn.model").string());
        out << meta.dump();
    }

    ScopedEnvironment<std::string> scoped_db(MIOPEN_SYSTEM_DB_PATH, dir.string());
    EXPECT_THROW(miopen::ai::immed_mode::MetadataND("gfx942", 2), miopen::Exception);

    fs::remove_all(dir);
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

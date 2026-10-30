/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// The rocKE backend's shipped kernel selection models (the selection models of
// HIPTENSOR_ROCKE_SELECT_MODEL_DIR, <target>/<cell>.model, which the build writes into the packs),
// read and walked by the library's own code (rocke_select_model.cpp), against the exporter
// (select/tools/train_select.py): every model loads, and a model with a <cell>.check file
// (written by the exporter next to every model of format v3: feature vectors of its training rows,
// each with the exported model's prediction by the exporter's walker and LightGBM's prediction)
// predicts the first exactly and the second within the compact export's rounding, on every route
// the model scores. CPU only.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "backends/rocke/rocke_select_model.hpp"

namespace
{
    namespace fs = std::filesystem;
    namespace sm = hiptensor::rockebackend::selectmodel;

    // The compact export rounds leaf values to 9 significant digits (train_select.py export):
    // the shipped model's predictions stay this close to LightGBM's.
    constexpr double kLightGbmTolerance = 1e-6;

    std::string slurp(fs::path const& p)
    {
        std::ifstream      f(p, std::ios::binary);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    std::string stripped(std::string text)
    {
        while(!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
        {
            text.pop_back();
        }
        return text;
    }

    std::vector<fs::path> shippedModels()
    {
        std::vector<fs::path> out;
        fs::path const        dir(HIPTENSOR_ROCKE_SELECT_MODEL_DIR);
        if(!fs::is_directory(dir))
        {
            return out;
        }
        for(auto const& target : fs::directory_iterator(dir))
        {
            if(!target.is_directory())
            {
                continue;
            }
            for(auto const& f : fs::directory_iterator(target.path()))
            {
                if(f.path().extension() == ".model")
                {
                    out.push_back(f.path());
                }
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
} // namespace

// Every shipped model is one the library uses (format, schema, features, trees).
TEST(RockeSelectModels, ShippedModelsLoad)
{
    auto const models = shippedModels();
    ASSERT_FALSE(models.empty()) << "no model in " << HIPTENSOR_ROCKE_SELECT_MODEL_DIR;
    for(auto const& path : models)
    {
        sm::Model   model;
        std::string why;
        EXPECT_TRUE(sm::parseModel(path.stem().string(), stripped(slurp(path)), model, why))
            << path << ": " << why;
        EXPECT_FALSE(model.trees.empty()) << path;
    }
}

// The library's walk of each shipped model of format v3 gives the exporter's predictions on its
// check vectors: the exported model's exactly, LightGBM's within the compact export's rounding;
// the vectors cover every route the model scores.
TEST(RockeSelectModels, ShippedModelsScoreLikeTheExporter)
{
    auto const& names  = sm::featureNames();
    int         models = 0, rows = 0;
    for(auto const& path : shippedModels())
    {
        SCOPED_TRACE(path.string());
        sm::Model   model;
        std::string why;
        auto const  text = stripped(slurp(path));
        ASSERT_TRUE(sm::parseModel(path.stem().string(), text, model, why)) << why;
        auto check = path;
        check.replace_extension(".check");
        if(model.format < 3)
        {
            // earlier formats were shipped without check vectors
            continue;
        }
        ASSERT_TRUE(fs::exists(check)) << "a v3 model ships with its check vectors";
        std::ifstream in(check);
        std::string   line;
        ASSERT_TRUE(std::getline(in, line));
        ASSERT_EQ(line.rfind("# features ", 0), 0u) << line;
        // the vectors' feature names: indices of the library's schema
        std::vector<size_t> index;
        {
            std::stringstream list(line.substr(11));
            std::string       name;
            while(std::getline(list, name, ','))
            {
                auto it = std::find_if(names.begin(), names.end(), [&name](char const* n) {
                    return name == n;
                });
                ASSERT_NE(it, names.end()) << "feature " << name;
                index.push_back(size_t(it - names.begin()));
            }
        }
        size_t const   routeAt = size_t(std::find_if(names.begin(),
                                                   names.end(),
                                                   [](char const* n) {
                                                       return std::string(n) == "route";
                                                   })
                                      - names.begin());
        std::set<int>  routes;
        int            n = 0;
        while(std::getline(in, line))
        {
            if(line.empty())
            {
                continue;
            }
            std::istringstream fields(line);
            std::string        walkerText, lightgbmText;
            ASSERT_TRUE(fields >> walkerText >> lightgbmText) << line;
            double const         walker   = std::strtod(walkerText.c_str(), nullptr);
            double const         lightgbm = std::strtod(lightgbmText.c_str(), nullptr);
            std::vector<int64_t> x(names.size(), 0);
            for(auto i : index)
            {
                long long v = 0;
                ASSERT_TRUE(fields >> v) << line;
                x[i] = int64_t(v);
            }
            double const got = model.predict(x);
            EXPECT_EQ(got, walker) << "row " << n;
            EXPECT_LE(std::fabs(got - lightgbm), kLightGbmTolerance) << "row " << n;
            if(std::find(index.begin(), index.end(), routeAt) != index.end())
            {
                routes.insert(int(x[routeAt]));
            }
            n++;
        }
        EXPECT_GE(n, 32) << "check vectors";
        // the vectors cover every route the model scores
        for(size_t r = 0; r < sm::kRoutes; r++)
        {
            if(model.scores(int(r)))
            {
                EXPECT_TRUE(routes.count(int(r))) << "no check vector of route " << r;
            }
        }
        models++;
        rows += n;
    }
    std::printf("checked %d models of format v3, %d vectors\n", models, rows);
}

// The model format v3: the digest of the planner's GEMM selection models a model names.
TEST(RockeSelectModels, FormatV3Header)
{
    auto const tree  = std::string("3;0 0;7.5 8.5;-1 -2;1 -3;1 0 1");
    auto       parse = [&](std::string const& head, sm::Model& m, std::string& why) {
        return sm::parseModel("contraction.h", head + "|" + tree, m, why);
    };
    sm::Model   m;
    std::string why;
    ASSERT_TRUE(parse("v3;schema=3;routes=3,8;margin=0.25;gemm=-;features=route;trees=1", m, why))
        << why;
    EXPECT_EQ(m.format, 3);
    EXPECT_EQ(m.schema, 3);
    EXPECT_TRUE(m.gemm.empty());
    EXPECT_FALSE(m.readsPrediction);
    EXPECT_DOUBLE_EQ(m.margin, 0.25);
    EXPECT_TRUE(m.scores(3) && m.scores(8) && !m.scores(7));
    std::string const digest(64, 'c');
    sm::Model         p;
    ASSERT_TRUE(parse("v3;schema=3;routes=11;margin=0;gemm=" + digest
                          + ";features=route,lg_pred_ns;trees=1",
                      p,
                      why))
        << why;
    EXPECT_EQ(p.gemm, digest);
    EXPECT_TRUE(p.readsPrediction);
    // an earlier schema with format v3
    sm::Model two;
    ASSERT_TRUE(parse("v3;schema=2;routes=11;margin=0;gemm=-;features=route;trees=1", two, why))
        << why;
    EXPECT_EQ(two.schema, 2);
    // v2 and v1 keep their meaning
    sm::Model v2, v1;
    ASSERT_TRUE(parse("v2;schema=2;routes=3,8;margin=0;features=route;trees=1", v2, why)) << why;
    EXPECT_EQ(v2.format, 2);
    EXPECT_TRUE(v2.gemm.empty());
    ASSERT_TRUE(parse("v1;schema=1;features=route;trees=1", v1, why)) << why;
    EXPECT_EQ(v1.format, 1);
    EXPECT_TRUE(v1.scores(3) && !v1.scores(7) && !v1.scores(8) && !v1.scores(11));
    // malformed headers are not used
    for(auto const& head :
        {std::string("v3;schema=3;routes=3;margin=0;gemm=abc;features=route;trees=1"),
         std::string("v3;schema=3;routes=3;margin=0;gemm=" + std::string(64, 'G')
                     + ";features=route;trees=1"),
         std::string("v3;schema=3;routes=3;margin=0;gemm=" + std::string(63, 'c')
                     + ";features=route;trees=1"),
         std::string("v3;schema=3;routes=3;margin=0;features=route;trees=1"),
         std::string("v3;schema=3;routes=3;gemm=-;margin=0;features=route;trees=1"),
         std::string("v3;schema=4;routes=3;margin=0;gemm=-;features=route;trees=1")})
    {
        sm::Model bad;
        EXPECT_FALSE(parse(head, bad, why)) << head;
    }
}

// CPU only: no device and no hipTensor handle (the shared test main needs a GPU).
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

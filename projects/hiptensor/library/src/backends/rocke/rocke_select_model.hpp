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

// The text format and the tree walk of the rocKE backend's selection models (rocke_select.hpp;
// select/README.md), without the library's runtime: the selection and the CPU test of the shipped
// models (test/00_unit/rocke_select_model_test.cpp) use the same code.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_MODEL_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_MODEL_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hiptensor
{
    namespace rockebackend
    {
        // Version of the feature schema. A model names the schema it was trained on and the
        // features it reads. Schemas are append only (schema 2 adds the JIT route's facts;
        // schema 3 gives the JIT route's candidates their bytes moved, lg_bytes and intensity_q8,
        // which schema 2 left 0, and adds jit_mode, whether the plan specializes its elementwise
        // launches with the JIT), so a model of an earlier schema is used, with the feature
        // values of its schema; one of a later schema is not.
        inline constexpr int kSelectSchema = 3;

        namespace selectmodel
        {
            // The names of the features of schema kSelectSchema, in the order of the library's
            // feature vectors.
            std::vector<char const*> const& featureNames();

            // Route values (rocke_plan.hpp Route) a model can name.
            constexpr size_t kRoutes = 32;

            // The routes a model of format v1 scores (the routes of the schema-1 corpus).
            bool legacyRoute(int route);

            // A gradient-boosted tree ensemble over named features. Node n of a tree goes left
            // when x[feature[n]] <= threshold[n]; a child c < 0 is leaf ~c.
            struct Tree
            {
                std::vector<int32_t> feature;
                std::vector<double>  threshold;
                std::vector<int32_t> left, right;
                std::vector<double>  leaf;
            };

            struct Model
            {
                std::string          id;
                std::vector<int32_t> featureIndex; // model feature -> schema index
                std::vector<Tree>    trees;
                // the routes the model was trained on (Route values), which it scores
                std::vector<bool> routes = std::vector<bool>(kRoutes, false);
                // the prediction by which the model's best candidate must beat the first scored
                // candidate of the static order to be selected instead of it (0: none)
                double margin = 0.0;
                // the model format (1, 2, 3) and the feature schema it was trained on
                int format = 1, schema = 1;
                // the digest of the planner's GEMM selection models its JIT candidates' features
                // were computed with (v3 gemm=<sha256 hex>; empty: "-", none)
                std::string gemm;
                // the model reads the planner's predicted time (lg_pred_ns)
                bool readsPrediction = false;

                // whether the model scores candidates of route `route` (a Route value)
                bool scores(int route) const
                {
                    return route >= 0 && size_t(route) < routes.size() && routes[size_t(route)];
                }

                // the prediction for a feature vector of the library's schema (featureNames())
                double predict(std::vector<int64_t> const& x) const;
            };

            // Parses a model text (see the formats in rocke_select_model.cpp parseModel); false
            // with `why` when it is not a model this library can use. `cell` names it in the
            // model's id ("<cell>@<hash of the text>").
            bool parseModel(std::string const& cell,
                            std::string const& text,
                            Model&             model,
                            std::string&       why);
        } // namespace selectmodel
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_MODEL_HPP

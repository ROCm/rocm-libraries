// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <Tensile/FixedLinearCatalogLibrary.hpp>
#include <Tensile/Serialization/Base.hpp>
namespace TensileLite { namespace Serialization {
template <typename MyProblem,typename MySolution,typename IO> struct MappingTraits<FixedLinearCatalogLibrary<MyProblem,MySolution>,IO>{using Library=FixedLinearCatalogLibrary<MyProblem,MySolution>;using iot=IOTraits<IO>;static void mapping(IO&io,Library&lib){auto ctx=static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));std::vector<int> indices;if(iot::outputting(io)){for(auto const&x:lib.solutions)indices.push_back(x.first);}iot::mapRequired(io,"table",indices);iot::mapRequired(io,"model_id",lib.modelId);iot::mapRequired(io,"feature_schema",lib.featureSchema);iot::mapRequired(io,"weights",lib.weights);iot::mapRequired(io,"cu_count",lib.cuCount);if(!iot::outputting(io)){if(!ctx||!ctx->solutions)iot::setError(io,"FixedLinearCatalog requires solution context.");for(int index:indices){auto it=ctx->solutions->find(index);if(it==ctx->solutions->end())iot::setError(io,"FixedLinearCatalog invalid solution index.");else lib.solutions.emplace_back(index,it->second);}if(lib.weights.size()!=22)iot::setError(io,"FixedLinearCatalog requires 22 weights.");}}const static bool flow=false;};
}}

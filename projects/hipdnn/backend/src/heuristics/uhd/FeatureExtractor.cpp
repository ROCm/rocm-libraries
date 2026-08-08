// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FeatureExtractor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

// Minimal SHA-256 implementation (no external dependencies)
// Based on RFC 6234 / FIPS 180-4
namespace
{

constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

constexpr uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }

constexpr uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

constexpr uint32_t sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }

constexpr uint32_t sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }

constexpr uint32_t gamma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }

constexpr uint32_t gamma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

std::string sha256(const std::string& input)
{
    std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    // Pre-processing: padding
    std::vector<uint8_t> msg(input.begin(), input.end());
    const uint64_t bitLen = msg.size() * 8;
    msg.push_back(0x80);
    while((msg.size() % 64) != 56)
    {
        msg.push_back(0x00);
    }
    for(int i = 7; i >= 0; --i)
    {
        msg.push_back(static_cast<uint8_t>((bitLen >> (i * 8)) & 0xFF));
    }

    // Process each 512-bit chunk
    for(size_t chunk = 0; chunk < msg.size(); chunk += 64)
    {
        std::array<uint32_t, 64> w{};
        for(size_t i = 0; i < 16; ++i)
        {
            w[i] = (static_cast<uint32_t>(msg[chunk + i * 4]) << 24) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 3]));
        }
        for(size_t i = 16; i < 64; ++i)
        {
            w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
        }

        auto a = h[0];
        auto b = h[1];
        auto c = h[2];
        auto d = h[3];
        auto e = h[4];
        auto f = h[5];
        auto g = h[6];
        auto hh = h[7];

        for(size_t i = 0; i < 64; ++i)
        {
            const uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
            const uint32_t t2 = sigma0(a) + maj(a, b, c);
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(auto hv : h)
    {
        oss << std::setw(8) << hv;
    }
    return oss.str();
}

} // anonymous namespace

namespace hipdnn_backend::heuristics::uhd
{

// ============================================================================
// FeatureExtractionContext
// ============================================================================

void FeatureExtractionContext::bindDeviceVars(const ValueMap& props)
{
    _ctx.bindNamespace("device", props);
}

void FeatureExtractionContext::bindKernelVars(const ValueMap& metadata)
{
    _ctx.bindNamespace("kernel", metadata);
}

void FeatureExtractionContext::bindQueryVars(const ValueMap& queryProps)
{
    _ctx.bindNamespace("q", queryProps);
}

void FeatureExtractionContext::bind(const std::string& fullName, VariableContext::ValueType value)
{
    _ctx.bind(fullName, std::move(value));
}

void FeatureExtractionContext::clear()
{
    _ctx.clear();
}

bool FeatureExtractionContext::hasAllVars(const std::unordered_set<std::string>& required) const
{
    for(const auto& var : required)
    {
        if(!_ctx.has(var))
        {
            return false;
        }
    }
    return true;
}

std::vector<std::string>
    FeatureExtractionContext::getMissingVars(const std::unordered_set<std::string>& required) const
{
    std::vector<std::string> missing;
    for(const auto& var : required)
    {
        if(!_ctx.has(var))
        {
            missing.push_back(var);
        }
    }
    return missing;
}

// ============================================================================
// FeatureExtractor
// ============================================================================

namespace
{

/// Largest numeric literal magnitude the cross-language canonical form is safe for.
///
/// Above this the two JSON writers stop agreeing, in three independent ways:
///   - nlohmann switches a double to scientific notation at 1e15, Python's repr at
///     1e16, so the whole decade renders differently ("1e+15" vs "1000000000000000.0");
///   - an integer outside int64/uint64 degrades to double in nlohmann while Python
///     keeps arbitrary precision -- and the lossy conversion makes distinct values
///     collide, so C++ would accept a hash computed over a *different* signature;
///   - non-finite values are a Python json extension that nlohmann rejects outright.
///
/// Feature literals are tile sizes, dimensions and thresholds, so this bound is many
/// orders of magnitude clear of anything real. Rejecting is the honest option: a
/// signature we cannot canonicalize identically on both sides has no usable hash.
constexpr double kMaxSafeNumericLiteral = 1e15;

void validateNumericLiterals(const nlohmann::json& node)
{
    if(node.is_number())
    {
        const double value = node.get<double>();
        if(!std::isfinite(value))
        {
            throw JsonLogicError("features_signature contains a non-finite numeric literal");
        }
        if(std::abs(value) >= kMaxSafeNumericLiteral)
        {
            throw JsonLogicError(
                "features_signature contains the numeric literal " + node.dump() +
                ", whose magnitude is at or above 1e15. The generator and the runtime "
                "render such values differently, so the features_hash would not match. "
                "Rescale the feature instead.");
        }
        return;
    }

    if(node.is_array())
    {
        for(const auto& element : node)
        {
            validateNumericLiterals(element);
        }
        return;
    }

    if(node.is_object())
    {
        for(const auto& item : node.items())
        {
            validateNumericLiterals(item.value());
        }
    }
}

} // namespace

nlohmann::json FeatureExtractor::parseSignatureEntry(const std::string& entry)
{
    // A bare reference such as `$q.seqlen_q` is the RFC 0019 §7.2 canonical spelling
    // and what tools/uhd_gen emits, but it is not valid JSON. Lift it to a JSON string
    // so it reaches the evaluator's variable branch.
    if(!entry.empty() && entry.front() == '$')
    {
        return nlohmann::json(entry);
    }

    return JsonLogicEvaluator::parse(entry);
}

FeatureExtractor::FeatureExtractor(const std::vector<std::string>& signature)
{
    _parsedExprs.reserve(signature.size());

    for(const auto& exprStr : signature)
    {
        auto parsed = parseSignatureEntry(exprStr);
        _parsedExprs.push_back(std::move(parsed));

        // Extract variable references from this expression
        auto vars = JsonLogicEvaluator::extractVariables(_parsedExprs.back());
        _varRefs.insert(vars.begin(), vars.end());
    }

    _signatureHash = computeHash(signature);
}

std::vector<double> FeatureExtractor::extract(const FeatureExtractionContext& ctx) const
{
    std::vector<double> features;
    features.reserve(_parsedExprs.size());

    for(const auto& expr : _parsedExprs)
    {
        features.push_back(_evaluator.evaluateDouble(expr, ctx.getContext()));
    }

    return features;
}

bool FeatureExtractor::validateContext(const FeatureExtractionContext& ctx) const
{
    return ctx.hasAllVars(_varRefs);
}

std::vector<std::string>
    FeatureExtractor::getMissingVariables(const FeatureExtractionContext& ctx) const
{
    return ctx.getMissingVars(_varRefs);
}

std::string FeatureExtractor::computeHash(const std::vector<std::string>& signature)
{
    // Canonical form is the parsed signature dumped as compact JSON, matching Python's
    // json.dumps(signature, separators=(",", ":")) in tools/uhd_gen. Parsing first means
    // the bare and pre-quoted spellings of a reference collapse to the same node, and
    // nlohmann handles escaping. Order is preserved deliberately: RFC 0019 §7.2 requires
    // the signature to match training exactly, so a permuted signature must hash
    // differently.
    nlohmann::json canonical = nlohmann::json::array();
    for(const auto& entry : signature)
    {
        canonical.push_back(parseSignatureEntry(entry));
    }

    // Reject literals the two languages would render differently before hashing them.
    validateNumericLiterals(canonical);

    std::string serialized;
    try
    {
        serialized = canonical.dump();
    }
    catch(const nlohmann::json::exception& e)
    {
        // dump() rejects invalid UTF-8 (type_error.316). That path never went through
        // parse() -- a bare "$ref" is lifted straight to a JSON string -- so it has to
        // be converted here or it escapes as a raw nlohmann type.
        throw JsonLogicError("features_signature cannot be serialized: " + std::string(e.what()));
    }

    const std::string fullHash = sha256(serialized);

    // Return first 16 chars with sha256: prefix to match Python format
    return "sha256:" + fullHash.substr(0, 16);
}

bool FeatureExtractor::validateAgainstKmdFields(
    const std::unordered_set<std::string>& kmdFieldNames) const
{
    const std::string kernelPrefix = "$kernel.";

    for(const auto& var : _varRefs)
    {
        if(var.rfind(kernelPrefix, 0) == 0)
        {
            // Extract field name after "$kernel."
            const std::string fieldName = var.substr(kernelPrefix.length());
            if(kmdFieldNames.find(fieldName) == kmdFieldNames.end())
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::string>
    FeatureExtractor::getMissingKmdFields(const std::unordered_set<std::string>& kmdFieldNames) const
{
    const std::string kernelPrefix = "$kernel.";
    std::vector<std::string> missing;

    for(const auto& var : _varRefs)
    {
        if(var.rfind(kernelPrefix, 0) == 0)
        {
            const std::string fieldName = var.substr(kernelPrefix.length());
            if(kmdFieldNames.find(fieldName) == kmdFieldNames.end())
            {
                missing.push_back(fieldName);
            }
        }
    }
    return missing;
}

} // namespace hipdnn_backend::heuristics::uhd

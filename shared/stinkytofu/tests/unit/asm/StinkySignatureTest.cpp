/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */
#include <gtest/gtest.h>

#include <array>
#include <string>

#include "stinkytofu/ir/asm/StinkySignature.hpp"

using namespace stinkytofu;

// ---------------------------------------------------------------------------
// Helper function tests
// ---------------------------------------------------------------------------

TEST(StinkySignatureHelpers, IsaVersionToGfx) {
    EXPECT_EQ(isaVersionToGfx({12, 5, 0}), "gfx1250");
    EXPECT_EQ(isaVersionToGfx({9, 0, 10}), "gfx90a");  // digits, not hex
    EXPECT_EQ(isaVersionToGfx({11, 0, 0}), "gfx1100");
}

TEST(StinkySignatureHelpers, ToHex) {
    EXPECT_EQ(toHex(0), "0");
    EXPECT_EQ(toHex(255), "ff");
    EXPECT_EQ(toHex(0xDEADBEEF), "deadbeef");
}

TEST(StinkySignatureHelpers, FieldDesc) {
    std::string d = fieldDesc("stride", 4, 14);
    EXPECT_NE(d.find("stride"), std::string::npos);
    EXPECT_NE(d.find("4"), std::string::npos);

    std::string d2 = fieldDesc("type", 2);
    EXPECT_NE(d2.find("type"), std::string::npos);
}

// ---------------------------------------------------------------------------
// createSrdUpperValue factory
// ---------------------------------------------------------------------------

TEST(StinkySignatureHelpers, CreateSrdUpperValueGfx9) {
    auto val = createSrdUpperValue({9, 0, 10});
    ASSERT_NE(val, nullptr);
    EXPECT_FALSE(val->toString().empty());
    EXPECT_FALSE(val->desc().empty());
    EXPECT_FALSE(val->fieldsDesc().empty());
}

TEST(StinkySignatureHelpers, CreateSrdUpperValueGfx10) {
    auto val = createSrdUpperValue({10, 3, 0});
    ASSERT_NE(val, nullptr);
    EXPECT_FALSE(val->toString().empty());
}

TEST(StinkySignatureHelpers, CreateSrdUpperValueGfx11) {
    auto val = createSrdUpperValue({11, 0, 0});
    ASSERT_NE(val, nullptr);
    EXPECT_FALSE(val->toString().empty());
}

TEST(StinkySignatureHelpers, CreateSrdUpperValueGfx12) {
    auto val = createSrdUpperValue({12, 0, 0});
    ASSERT_NE(val, nullptr);
    EXPECT_FALSE(val->toString().empty());
}

TEST(StinkySignatureHelpers, CreateSrdUpperValueGfx1250) {
    auto val = createSrdUpperValue({12, 5, 0});
    ASSERT_NE(val, nullptr);
    EXPECT_FALSE(val->toString().empty());
}

// ---------------------------------------------------------------------------
// SignatureKernelDescriptor
// ---------------------------------------------------------------------------

TEST(SignatureKernelDescriptorTest, Construction) {
    SignatureKernelDescriptor kd("my_kernel", {12, 5, 0}, 0, {1, 1, 1}, 1, 64, 128, 0, 64);
    EXPECT_EQ(kd.kernelName, "my_kernel");
    EXPECT_EQ(kd.wavefrontSize, 64);
    EXPECT_EQ(kd.getNextFreeVgpr(), 128);
}

TEST(SignatureKernelDescriptorTest, SetGprs) {
    SignatureKernelDescriptor kd("k", {12, 5, 0}, 0, {1, 1, 1}, 1);
    kd.setGprs(256, 0, 64);
    EXPECT_EQ(kd.getNextFreeVgpr(), 256);
    EXPECT_EQ(kd.getNextFreeSgpr(), 64);
}

TEST(SignatureKernelDescriptorTest, ToStringContainsKernelName) {
    SignatureKernelDescriptor kd("my_kernel", {12, 5, 0}, 0, {1, 1, 1}, 1);
    std::string s = kd.toString();
    EXPECT_NE(s.find("my_kernel"), std::string::npos);
}

// ---------------------------------------------------------------------------
// SignatureCodeMeta
// ---------------------------------------------------------------------------

TEST(SignatureCodeMetaTest, AddArgAndToString) {
    SignatureCodeMeta cm("kern", 2, 0, 256, 64, "v5", 128, 64);
    cm.addArg("buf", SignatureValueKind::SIG_GLOBALBUFFER, "struct", "global");
    cm.addArg("alpha", SignatureValueKind::SIG_VALUE, "f32");
    std::string s = cm.toString();
    EXPECT_NE(s.find("kern"), std::string::npos);
    EXPECT_NE(s.find("buf"), std::string::npos);
}

// ---------------------------------------------------------------------------
// SignatureBase
// ---------------------------------------------------------------------------

TEST(SignatureBaseTest, Construction) {
    SignatureBase sig("my_kernel", {12, 5, 0}, 2, "v5", 0, {1, 1, 1}, 1, 256, 64, 128, 0, 64);
    EXPECT_EQ(sig.getNextFreeVgpr(), 128);
}

TEST(SignatureBaseTest, SetGprsUpdatesDescriptorAndMeta) {
    SignatureBase sig("k", {12, 5, 0}, 2, "v5", 0, {1, 1, 1}, 1, 256);
    sig.setGprs(200, 0, 80);
    EXPECT_EQ(sig.getNextFreeVgpr(), 200);
    EXPECT_EQ(sig.getNextFreeSgpr(), 80);
}

TEST(SignatureBaseTest, AddArgAndToStringContainsArg) {
    SignatureBase sig("k", {12, 5, 0}, 2, "v5", 0, {1, 1, 1}, 1, 256);
    sig.addArg("inputA", SignatureValueKind::SIG_GLOBALBUFFER, "struct", "global");
    std::string s = sig.toString();
    EXPECT_NE(s.find("inputA"), std::string::npos);
}

TEST(SignatureBaseTest, ToStringContainsGfxTarget) {
    SignatureBase sig("k", {12, 5, 0}, 2, "v5", 0, {1, 1, 1}, 1, 256);
    std::string s = sig.toString();
    EXPECT_NE(s.find("gfx1250"), std::string::npos);
}

TEST(SignatureBaseTest, DescriptionMethods) {
    SignatureBase sig("k", {12, 5, 0}, 2, "v5", 0, {1, 1, 1}, 1, 256);
    sig.addDescriptionTopic("Test topic");
    sig.addDescriptionBlock("Block text");
    sig.addDescription("Extra info");
    std::string s = sig.toString();
    EXPECT_FALSE(s.empty());
    sig.clearDescription();
}

TEST(SignatureBaseTest, Block3Line) {
    std::string s = SignatureBase::block3Line("hello world");
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("hello world"), std::string::npos);
}

// ---------------------------------------------------------------------------
// KernelBody
// ---------------------------------------------------------------------------

TEST(KernelBodyTest, ConstructionAndGetName) {
    KernelBody kb("my_kernel");
    EXPECT_EQ(kb.getName(), "my_kernel");
    EXPECT_EQ(kb.getNextFreeVgpr(), 0);
    EXPECT_EQ(kb.getNextFreeSgpr(), 0);
}

TEST(KernelBodyTest, AddSignatureAndSetGprs) {
    KernelBody kb("k");
    auto sig = std::make_shared<SignatureBase>("k", std::array<int, 3>{12, 5, 0}, 2, "v5", 0,
                                               std::array<int, 3>{1, 1, 1}, 1, 256, 64);
    kb.addSignature(sig);
    kb.setGprs(128, 0, 64);
    EXPECT_EQ(kb.getNextFreeVgpr(), 128);
    EXPECT_EQ(kb.getNextFreeSgpr(), 64);
    EXPECT_EQ(kb.getSignature(), sig);
}

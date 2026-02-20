// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FUSE_SHIM_H
#define FUSE_SHIM_H

#include "tree_node.h"

/*****************************************************
 * TR- transpose + FFT: possible fusable Node  *
 *****************************************************/
class TRFuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    TRFuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * RT- FFT + transpose: possible fusable Node  *
 *****************************************************/
class RTFuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    RTFuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * R_T-ZXY = FFT + transpose_Z_XY -> SBRCTransZ_XYNode *
 *****************************************************/
class RT_ZXY_FuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    RT_ZXY_FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * R_T-XYZ = FFT + transpose_XY_Z -> SBRCTransXY_ZNode *
 *****************************************************/
class RT_XYZ_FuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    RT_XYZ_FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * R2C_Trans = REAL_2_CMPLX + TRANSPOSE*
 *****************************************************/
class R2CTrans_FuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    R2CTrans_FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    bool PlacementFusable(OperatingBuffer iBuf,
                          OperatingBuffer firstOBuf,
                          OperatingBuffer lastOBuf) override;

    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * Trans_C2R = TRANSPOSE + CMPLX_2_REAL *
 *****************************************************/
class TransC2R_FuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    TransC2R_FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

/*****************************************************
 * STK_R2C_Trans = STOCKHAM + REAL_2_CMPLX + TRANSPOSE*
 *****************************************************/
class STK_R2CTrans_FuseShim : public FuseShim
{
    friend class NodeFactory;

protected:
    STK_R2CTrans_FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : FuseShim(components, type)
    {
        allowInplace  = false;
        schemeFusable = CheckSchemeFusable();
    }

    bool CheckSchemeFusable() override;

public:
    std::unique_ptr<TreeNode> FuseKernels() override;
};

#endif // FUSE_SHIM_H

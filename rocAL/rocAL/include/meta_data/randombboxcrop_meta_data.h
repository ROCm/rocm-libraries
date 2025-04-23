/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pipeline/commons.h"

struct CropCord {
    CropCord() {
        crop_left = 0;
        crop_top = 0;
        crop_right = 0;
        crop_bottom = 0;
    }
    CropCord(float l, float t, float r, float b) {
        crop_left = l;
        crop_top = t;
        crop_right = r;
        crop_bottom = b;
    }
    float crop_left;
    float crop_top;
    float crop_right;
    float crop_bottom;
};

using pCropCord = std::shared_ptr<CropCord>;

struct CropCordBatch {
    virtual ~CropCordBatch() = default;
    void clear() {
        _crop_cords.clear();
    }
    void resize(int batch_size) {
        _crop_cords.resize(batch_size);
    }
    int size() {
        return _crop_cords.size();
    }
    CropCordBatch& operator+=(CropCordBatch& other) {
        _crop_cords.insert(_crop_cords.end(), other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        return *this;
    }
    CropCordBatch* concatenate(CropCordBatch* other) {
        *this += *other;
        return this;
    }
    std::shared_ptr<CropCordBatch> clone() {
        return std::make_shared<CropCordBatch>(*this);
    }
    std::vector<pCropCord>& get_bb_cords_batch() { return _crop_cords; }

   protected:
    std::vector<pCropCord> _crop_cords = {};
};

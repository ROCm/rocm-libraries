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

#include "meta_data/randombboxcrop_meta_data_reader_factory.h"

#include <memory>

#include "pipeline/exception.h"

std::shared_ptr<RandomBBoxCrop_MetaDataReader> create_meta_data_reader(const RandomBBoxCrop_MetaDataConfig& config, std::shared_ptr<CropCordBatch>& meta_data_batch) {
    switch (config.reader_type()) {
        case RandomBBoxCrop_MetaDataReaderType::RandomBBoxCropReader: {
            if (config.type() != RandomBBoxCrop_MetaDataType::BoundingBox)
                THROW("RANDOMBBOXCROP can only be used to load CROP OUTPUTS")
            auto ret = std::make_shared<RandomBBoxCropReader>();
            meta_data_batch = std::make_shared<CropCordBatch>();
            ret->init(config, meta_data_batch);
            return ret;
        } break;
        default:
            THROW("RandomBBoxCrop_MetaDataReader type is unsupported : " + TOSTR(config.reader_type()));
    }
}

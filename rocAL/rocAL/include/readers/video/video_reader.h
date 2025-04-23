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
#include <tuple>
#include <vector>

#include "pipeline/commons.h"
#include "meta_data/meta_data_reader.h"
#include "readers/image/image_reader.h"
#include "readers/video/video_properties.h"

#ifdef ROCAL_VIDEO
struct SequenceInfo {
    size_t start_frame_number;
    std::string video_file_name;
};

class VideoReader {
   public:
    enum class Status {
        OK = 0
    };
    //! Looks up the folder which contains the files, amd loads the video sequences
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    virtual Status initialize(ReaderConfig desc) = 0;

    //! Reads the next resource item
    virtual SequenceInfo get_sequence_info() = 0;

    //! Resets the object's state to read from the first file in the folder
    virtual void reset() = 0;

    //! Returns the name of the latest file opened
    virtual std::string id() = 0;

    //! Returns the number of items remained in this resource
    virtual unsigned count_items() = 0;

    virtual ~VideoReader() = default;
};
#endif

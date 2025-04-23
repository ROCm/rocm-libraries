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

#include <VX/vx.h>
#include <VX/vx_types.h>

#include <array>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>
#if ENABLE_HIP
#include "device/device_manager_hip.h"
#include "hip/hip_runtime.h"
#elif ENABLE_OPENCL
#include "device/device_manager.h"
#endif
#include "pipeline/commons.h"

/*! \brief Converts Rocal Memory type to OpenVX memory type
 *
 * @param mem input Rocal type
 * @return the OpenVX type associated with input argument
 */
vx_enum vx_image_mem_type(RocalMemType mem);
struct Point {
    unsigned x;  // Along the width
    unsigned y;  // Along the height
};

struct ROI {
    Point p1;  // Upper left
    Point p2;  // bottom right
};

// +-----------------------------------------> X direction
// |  ___________________________________
// |  |   p1(x,y)      |                |
// |  |    +-----------|-----------+    |
// |  |    |           |           |    |
// |  -----------------o-----------------
// |  |    |           |           |    |
// |  |    +-----------|-----------+    |
// |  |                |        p2(x,y) |
// |  +++++++++++++++++++++++++++++++++++
// |
// V Y directoin

/*! \brief Holds the information about an OpenVX image */

struct ImageInfo {
    friend struct Image;
    enum class Type {
        UNKNOWN = -1,
        REGULAR = 0,
        VIRTUAL = 1,
        HANDLE = 2
    };
    //! Default constructor,
    /*! initializes memory type to host and batch size to 1 */
    ImageInfo();

    //! Initializer constructor
    ImageInfo(
        unsigned width,
        unsigned height,
        unsigned batch_size,
        unsigned color_planes_count,
        RocalMemType mem_type,
        RocalColorFormat color_format);

    unsigned width() const { return _width; }
    unsigned height_batch() const { return _height * _batch_size; }
    unsigned height_single() const { return _height; }
    unsigned color_plane_count() const { return _color_planes; }
    void width(unsigned width) { _width = width; }
    void height(unsigned height) { _height = height; }
    Type type() const { return _type; }
    unsigned batch_size() const { return _batch_size; }
    RocalMemType mem_type() const { return _mem_type; }
    uint64_t data_size() const { return _data_size; }
    RocalColorFormat color_format() const { return _color_fmt; }
    unsigned get_roi_width(int image_batch_idx) const;
    unsigned get_roi_height(int image_batch_idx) const;
    uint32_t* get_roi_width() const;
    uint32_t* get_roi_height() const;
    const std::vector<uint32_t>& get_roi_width_vec() const;
    const std::vector<uint32_t>& get_roi_height_vec() const;
    const std::vector<uint32_t>& get_original_width_vec() const;
    const std::vector<uint32_t>& get_original_height_vec() const;

   private:
    Type _type = Type::UNKNOWN;                               //!< image type, whether is virtual image, created from handle or is a regular image
    unsigned _width;                                          //!< image width for a single image in the batch
    unsigned _height;                                         //!< image height for a single image in the batch
    unsigned _color_planes;                                   //!< number of color planes
    unsigned _batch_size;                                     //!< the batch size (images in the batch are stacked on top of each other)
    uint64_t _data_size;                                      //!< total size of the memory needed to keep the image's data in bytes including all planes
    RocalMemType _mem_type;                                   //!< memory type, currently either OpenCL or Host
    RocalColorFormat _color_fmt;                              //!< color format of the image
    std::shared_ptr<std::vector<uint32_t>> _roi_width;        //!< The actual image width stored in the buffer, it's always smaller than _width/_batch_size. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one and observed for all.
    std::shared_ptr<std::vector<uint32_t>> _roi_height;       //!< The actual image height stored in the buffer, it's always smaller than _height. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one changes can be observed for all.
    std::shared_ptr<std::vector<uint32_t>> _original_width;   //!< The original image width decoded by the turbojpeg library before scaling to _roi_width
    std::shared_ptr<std::vector<uint32_t>> _original_height;  //!< The original image height decoded by the turbojpeg library before scaling to _roi_height

    void reallocate_image_roi_buffers();
};
bool operator==(const ImageInfo& rhs, const ImageInfo& lhs);

/*! \brief Holds an OpenVX image and it's info
 *
 * Keeps the information about the image that can be queried using OVX API as well,
 * but for simplicity and ease of use, they are kept in separate fields
 */
struct Image {
    int swap_handle(void* handle);

    const ImageInfo& info() { return _info; }
    //! Default constructor
    Image() = delete;
    void* buffer() { return _mem_handle; }
    vx_image handle() { return vx_handle; }
    vx_context context() { return _context; }
#if ENABLE_OPENCL
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#elif ENABLE_HIP
    unsigned copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync);
    unsigned copy_data(hipStream_t stream, void* hip_memory, bool sync);
#endif
    //! Default destructor
    /*! Releases the OpenVX image */
    ~Image();

    //! Constructor accepting the image information as input
    explicit Image(const ImageInfo& img_info);

    int create(vx_context context);
    void update_image_roi(const std::vector<uint32_t>& width, const std::vector<uint32_t>& height);
    void update_image_original_dims(const std::vector<uint32_t>& original_width, const std::vector<uint32_t>& original_height);
    void reset_image_roi() { _info.reallocate_image_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since image's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);
    bool is_handle_set() { return (vx_handle != 0); }

   private:
    vx_image vx_handle = nullptr;  //!< The OpenVX image
    void* _mem_handle = nullptr;   //!< Pointer to the image's internal buffer (opencl, host or hip)
    ImageInfo _info;               //!< The structure holding the info related to the stored OpenVX image
    vx_context _context = nullptr;
};

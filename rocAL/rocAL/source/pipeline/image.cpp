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
#include <cstdio>
#if ENABLE_OPENCL
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>

#include <cstring>

#include "pipeline/commons.h"
#include "pipeline/image.h"

vx_enum vx_image_mem_type(RocalMemType mem) {
    switch (mem) {
        case RocalMemType::OCL: {
            return VX_MEMORY_TYPE_OPENCL;
        } break;
        case RocalMemType::HOST: {
            return VX_MEMORY_TYPE_HOST;
        } break;
        case RocalMemType::HIP: {
            return VX_MEMORY_TYPE_HIP;
        } break;
        default:
            throw std::runtime_error("Memory type not valid");
    }
}
bool operator==(const ImageInfo& rhs, const ImageInfo& lhs) {
    return (rhs.width() == lhs.width() &&
            rhs.height_batch() == lhs.height_batch() &&
            rhs.mem_type() == lhs.mem_type() &&
            rhs.color_format() == lhs.color_format() &&
            rhs.color_plane_count() == lhs.color_plane_count());
}

uint32_t* ImageInfo::get_roi_width() const {
    return _roi_width->data();
}

uint32_t* ImageInfo::get_roi_height() const {
    return _roi_height->data();
}

const std::vector<uint32_t>& ImageInfo::get_roi_width_vec() const {
    return *_roi_width;
}

const std::vector<uint32_t>& ImageInfo::get_roi_height_vec() const {
    return *_roi_height;
}

const std::vector<uint32_t>& ImageInfo::get_original_width_vec() const {
    return *_original_width;
}

const std::vector<uint32_t>& ImageInfo::get_original_height_vec() const {
    return *_original_height;
}

unsigned ImageInfo::get_roi_width(int image_batch_idx) const {
    if ((unsigned)image_batch_idx >= _roi_width->size())
        THROW("Accesing image width out of batch size range")
    if (!_roi_width->at(image_batch_idx))
        THROW("Accessing uninitialized int parameter associated with image width")
    return _roi_width->at(image_batch_idx);
}

unsigned ImageInfo::get_roi_height(int image_batch_idx) const {
    if ((unsigned)image_batch_idx >= _roi_height->size())
        THROW("Accesing image height out of batch size range")
    if (!_roi_height->at(image_batch_idx))
        THROW("Accessing uninitialized int parameter associated with image height")
    return _roi_height->at(image_batch_idx);
}
void ImageInfo::reallocate_image_roi_buffers() {
    _roi_height = std::make_shared<std::vector<uint32_t>>(_batch_size);
    _roi_width = std::make_shared<std::vector<uint32_t>>(_batch_size);
    _original_height = std::make_shared<std::vector<uint32_t>>(_batch_size);
    _original_width = std::make_shared<std::vector<uint32_t>>(_batch_size);
    for (unsigned i = 0; i < _batch_size; i++) {
        _roi_height->at(i) = height_single();
        _roi_width->at(i) = width();
    }
}
ImageInfo::ImageInfo() : _type(Type::UNKNOWN),
                         _width(0),
                         _height(0),
                         _color_planes(1),
                         _batch_size(1),
                         _data_size(0),
                         _mem_type(RocalMemType::HOST),
                         _color_fmt(RocalColorFormat::U8) {}

ImageInfo::ImageInfo(
    unsigned width_,
    unsigned height_,
    unsigned batches,
    unsigned planes,
    RocalMemType mem_type_,
    RocalColorFormat col_fmt_) : _type(Type::UNKNOWN),
                                 _width(width_),
                                 _height(height_),
                                 _color_planes(planes),
                                 _batch_size(batches),
                                 _data_size((static_cast<uint64_t>(width_ * height_ * _batch_size * planes))),
                                 _mem_type(mem_type_),
                                 _color_fmt(col_fmt_) {
    // initializing each image dimension in the batch with the maximum image size, they'll get updated later during the runtime
    reallocate_image_roi_buffers();
}

void Image::update_image_roi(const std::vector<uint32_t>& width, const std::vector<uint32_t>& height) {
    if (width.size() != height.size())
        THROW("Batch size of image height and width info does not match")

    if (width.size() != info().batch_size())
        THROW("The batch size of actual image height and width different from image batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))
    if (!_info._roi_width || !_info._roi_height)
        THROW("ROI width or ROI height vector not created")
    for (unsigned i = 0; i < info().batch_size(); i++) {
        if (width[i] > _info.width()) {
            ERR("Given ROI width is larger than buffer width for image[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(_info.width()))
            _info._roi_width->at(i) = _info.width();
        } else {
            _info._roi_width->at(i) = width[i];
        }

        if (height[i] > _info.height_single()) {
            ERR("Given ROI height is larger than buffer height for image[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(_info.height_single()))
            _info._roi_height->at(i) = _info.height_single();
        } else {
            _info._roi_height->at(i) = height[i];
        }
    }
}

void Image::update_image_original_dims(const std::vector<uint32_t>& original_width, const std::vector<uint32_t>& original_height) {
    if (original_width.size() != original_height.size())
        THROW("Batch size of image height and width info does not match")

    if (original_width.size() != info().batch_size())
        THROW("The batch size of actual image height and width different from image batch size " + TOSTR(original_width.size()) + " != " + TOSTR(info().batch_size()))
    if (!_info._original_width || !_info._original_height)
        THROW("ROI width or ROI height vector not created")
    for (unsigned i = 0; i < info().batch_size(); i++) {
        _info._original_width->at(i) = original_width[i];
        _info._original_height->at(i) = original_height[i];
    }
}

Image::~Image() {
    vxReleaseImage(&vx_handle);
}

//! Converts the Rocal color format type to OpenVX
/*!
 * For OpenVX there is no BGR color format supported.
 * The input images loaded in GBR format should be
 * reordered before passed to OpenVX, and match RGB.
 */
vx_df_image interpret_color_fmt(RocalColorFormat color_format) {
    switch (color_format) {
        case RocalColorFormat::RGB24:
        case RocalColorFormat::BGR24:
        case RocalColorFormat::RGB_PLANAR:  // not theoretically correct, but keeping it for the same memory allocation
            return VX_DF_IMAGE_RGB;

        case RocalColorFormat::U8:
            return VX_DF_IMAGE_U8;

        default:
            THROW("Unsupported Image type " + TOSTR(color_format))
    }
}

Image::Image(const ImageInfo& img_info) : _info(img_info) {
    _info._type = ImageInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int Image::create_virtual(vx_context context, vx_graph graph) {
    if (vx_handle)
        return -1;

    _context = context;

    // create a virtual image as the output image for this node
    vx_handle = vxCreateVirtualImage(graph, _info.width(), _info.height_batch(), VX_DF_IMAGE_VIRT);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualImage(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))

    _info._type = ImageInfo::Type::VIRTUAL;
    return 0;
}

int Image::create_from_handle(vx_context context) {
    if (vx_handle) {
        WRN("Image object create method is already called ")
        return -1;
    }

    _context = context;

    // TODO: the pointer passed here changes if the number of planes are more than one
    vx_imagepatch_addressing_t addr_in = {0};
    void* ptr[1] = {nullptr};

    addr_in.step_x = 1;
    addr_in.step_y = 1;
    addr_in.scale_x = VX_SCALE_UNITY;
    addr_in.scale_y = VX_SCALE_UNITY;
    addr_in.dim_x = _info.width();
    addr_in.dim_y = _info.height_batch();

    vx_uint32 alignpixels = 32;

    addr_in.stride_x = _info._color_planes;

    if (alignpixels == 0)
        addr_in.stride_y = addr_in.dim_x * addr_in.stride_x;
    else
        addr_in.stride_y = ((addr_in.dim_x + alignpixels - 1) & ~(alignpixels - 1)) * addr_in.stride_x;

    if (_info.height_batch() == 0 || _info.width() == 0 || _info._color_planes == 0)
        THROW("Invalid image dimension " + TOSTR(_info.height_batch()) + " x " + TOSTR(_info.width()) + " x " + TOSTR(_info._color_planes));

    vx_status status;
    vx_size size = (addr_in.dim_y + 0) * (addr_in.stride_y + 0);

    vx_df_image vx_color_format = interpret_color_fmt(_info._color_fmt);
    vx_handle = vxCreateImageFromHandle(context, vx_color_format, &addr_in, ptr, vx_image_mem_type(_info._mem_type));
    if ((status = vxGetStatus((vx_reference)vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateImageFromHandle(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))

    _info._type = ImageInfo::Type::HANDLE;
    _info._data_size = size;
    return 0;
}
int Image::create(vx_context context) {
    if (vx_handle)
        return -1;

    _context = context;

    vx_status status;
    vx_df_image vx_color_format = interpret_color_fmt(_info._color_fmt);
    vx_handle = vxCreateImage(context, _info.width(), _info.height_batch(), vx_color_format);
    if ((status = vxGetStatus((vx_reference)vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateImage(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height_batch()) + "]): failed " + TOSTR(status))
    _info._type = ImageInfo::Type::REGULAR;
    return 0;
}

#if ENABLE_OPENCL
unsigned Image::copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync) {
    if (_info._type != ImageInfo::Type::HANDLE)
        return 0;

    unsigned size = _info.width() *
                    _info.height_batch() *
                    _info.color_plane_count();

    if (_info._mem_type == RocalMemType::OCL) {
        cl_int status;
        if ((status = clEnqueueReadBuffer(queue,
                                          (cl_mem)_mem_handle,
                                          sync ? (CL_TRUE) : CL_FALSE,
                                          0,
                                          size,
                                          user_buffer,
                                          0, nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))

    } else {
        memcpy(user_buffer, _mem_handle, size);
    }
    return size;
}

unsigned Image::copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync) {
    return 0;
}

#elif ENABLE_HIP
unsigned Image::copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync) {
    if (_info._type != ImageInfo::Type::HANDLE)
        return 0;

    unsigned size = _info.width() *
                    _info.height_batch() *
                    _info.color_plane_count();

    if (_info._mem_type == RocalMemType::HIP) {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoHAsync((void*)user_buffer, _mem_handle, size, stream)))
            THROW("copy_data::hipMemcpyDtoHAsync failed: " + TOSTR(status))
        if (sync) {
            if ((status = hipStreamSynchronize(stream)))
                THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
        }

    } else {
        memcpy(user_buffer, _mem_handle, size);
    }
    return size;
}
unsigned Image::copy_data(hipStream_t stream, void* hip_memory, bool sync) {
    // todo:: copy from host to device
    return 0;
}
#endif
int Image::swap_handle(void* handle) {
    vx_status vxstatus;
    void* ptr_in[] = {handle};
    if ((vxstatus = vxSwapImageHandle(vx_handle, ptr_in, nullptr, 1)) != VX_SUCCESS) {
        ERR("Swap handles failed " + TOSTR(vxstatus));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}
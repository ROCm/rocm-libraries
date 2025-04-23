/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "roc_pyvideodecode.h"
#include "colorspace_kernels.h"
#include "resize_kernels.h"

using namespace std;

void PyRocVideoDecoderInitializer(py::module& m) {
        py::class_<PyRocVideoDecoder> (m, "PyRocVideoDecoder")
        .def(py::init<int,int,rocDecVideoCodec,bool,const Rect *,int,int,uint32_t>(),
                    py::arg("device_id") = 0, py::arg("out_mem_type") = 0, py::arg("codec") = rocDecVideoCodec_HEVC, py::arg("force_zero_latency") = false, 
                    py::arg("p_crop_rect") = nullptr, py::arg("max_width") = 0, py::arg("max_height") = 0, py::arg("clk_rate") = 1000)
        .def("GetDeviceinfo",&PyRocVideoDecoder::PyGetDeviceinfo)
        .def("DecodeFrame",&PyRocVideoDecoder::PyDecodeFrame)
        .def("GetFrameYuv",&PyRocVideoDecoder::PyGetFrameYuv)
        .def("GetFrameRgb",&PyRocVideoDecoder::PyGetFrameRgb)
        .def("ResizeFrame",&PyRocVideoDecoder::PyResizeFrame)
        .def("GetWidth",&PyRocVideoDecoder::PyGetWidth)
        .def("GetHeight",&PyRocVideoDecoder::PyGetHeight)
        .def("GetStride",&PyRocVideoDecoder::PyGetStride)
        .def("GetFrameSize",&PyRocVideoDecoder::PyGetFrameSize)
        .def("SaveFrameToFile",&PyRocVideoDecoder::PySaveFrameToFile)
        .def("ReleaseFrame",&PyRocVideoDecoder::PyReleaseFrame)
        .def("GetOutputSurfaceInfo",&PyRocVideoDecoder::PyGetOutputSurfaceInfo)
        .def("GetResizedOutputSurfaceInfo",&PyRocVideoDecoder::PyGetResizedOutputSurfaceInfo)
        .def("GetNumOfFlushedFrames",&PyRocVideoDecoder::PyGetNumOfFlushedFrames)
        .def("SetReconfigParams",&PyRocVideoDecoder::PySetReconfigParams)
        .def("IsCodecSupported",&PyRocVideoDecoder::PyCodecSupported)
        .def("GetBitDepth",&PyRocVideoDecoder::PyGetBitDepth)
#if ROCDECODE_CHECK_VERSION(0,6,0)
        .def("AddDecoderSessionOverHead",&PyRocVideoDecoder::PyAddDecoderSessionOverHead)
        .def("GetDecoderSessionOverHead",&PyRocVideoDecoder::PyGetDecoderSessionOverHead)
#endif
    ;
}

// callback function to flush last frames and save it to file when reconfigure happens
// reference to non-static member function must be called
int PyReconfigureFlushCallback(void *p_viddec_obj, uint32_t flush_mode, void * p_user_struct) {
    int n_frames_flushed = 0;
    if ((p_viddec_obj == nullptr) ||  (p_user_struct == nullptr))
        return n_frames_flushed;
    RocVideoDecoder *viddec = static_cast<RocVideoDecoder *> (p_viddec_obj);
    OutputSurfaceInfo *surf_info;
    if (!viddec->GetOutputSurfaceInfo(&surf_info)) {
        std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
        return n_frames_flushed;
    }
    uint8_t *pframe = nullptr;
    int64_t pts;
    while ((pframe = viddec->GetFrame(&pts))) {
        if (flush_mode != RECONFIG_FLUSH_MODE_NONE) {
            if (flush_mode == ReconfigFlushMode::RECONFIG_FLUSH_MODE_DUMP_TO_FILE) {
                ReconfigDumpFileStruct *p_dump_file_struct = static_cast<ReconfigDumpFileStruct *>(p_user_struct);
                if (p_dump_file_struct->b_dump_frames_to_file) {
                    viddec->SaveFrameToFile(p_dump_file_struct->output_file_name, pframe, surf_info);
                }
            }
        }
        // release and flush frame
        viddec->ReleaseFrame(pts, true);
        n_frames_flushed ++;
    }
    return n_frames_flushed;
}

py::object PyRocVideoDecoder::PySetReconfigParams(int flush_mode, std::string& output_file_name_in) {
    ReconfigFlushMode mode = static_cast<ReconfigFlushMode>(flush_mode);
    if(!output_file_name_in.empty()) {
        PyReconfigDumpFileStruct.output_file_name = output_file_name_in;
        PyReconfigDumpFileStruct.b_dump_frames_to_file = true;
    } else {
        if(mode == RECONFIG_FLUSH_MODE_DUMP_TO_FILE)
            mode = RECONFIG_FLUSH_MODE_NONE;
    }
    PyReconfigParams.p_fn_reconfigure_flush = PyReconfigureFlushCallback;
    PyReconfigParams.p_reconfig_user_struct = &PyReconfigDumpFileStruct;
    PyReconfigParams.reconfig_flush_mode = mode;
    // set the parent class
    SetReconfigParams(&PyReconfigParams);
    return py::cast<py::none>(Py_None);
}

void PyRocVideoDecoder::InitConfigStructure() {
    // init config struct
    configInfo.reset(new ConfigInfo());
    configInfo.get()->device_name = std::string("");
    configInfo.get()->gcn_arch_name = std::string("");
    configInfo.get()->pci_bus_id = 0;
    configInfo.get()->pci_domain_id = 0;
    configInfo.get()->pci_device_id = 0;
    // init flush callback struct: support multi-resolution video streams
    PyReconfigDumpFileStruct.b_dump_frames_to_file = false;
    PyReconfigDumpFileStruct.output_file_name.clear();
    PyReconfigParams.p_fn_reconfigure_flush = nullptr;
    PyReconfigParams.p_reconfig_user_struct = nullptr;
    PyReconfigParams.reconfig_flush_mode = 0;
}

PyRocVideoDecoder::~PyRocVideoDecoder() {
    // free new RGB frame ptr if used
    if (frame_ptr_rgb != nullptr) {
        hipError_t hip_status = hipFree(frame_ptr_rgb);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
        }
        frame_ptr_rgb = nullptr;
    }
    if( post_process_class != nullptr ) {
        delete post_process_class;
        post_process_class = nullptr;
    }
    // free new resized YUV frame if used
    if (frame_ptr_resized != nullptr) {
        hipError_t hip_status = hipFree(frame_ptr_resized);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
        }
        frame_ptr_resized = nullptr;
    }
    // free new surface allocated locally
    if (resized_surf_info != nullptr) {
        free(resized_surf_info);
        resized_surf_info = nullptr;
    }
}

int PyRocVideoDecoder::PyDecodeFrame(PyPacketData& packet) {
    if(packet.bitstream_size == 0)
        packet.pkt_flags |= ROCDEC_PKT_ENDOFSTREAM;
    int decoded_frame_count = DecodeFrame(reinterpret_cast<const uint8_t *>(packet.bitstream_adrs), static_cast<size_t>(packet.bitstream_size), packet.pkt_flags, packet.frame_pts);
    return decoded_frame_count;
}

// for python binding
py::object PyRocVideoDecoder::PyGetFrameYuv(PyPacketData& packet, bool SeparateYuvPlanes) {
    int frame_size = GetFrameSize();
    int64_t pts = packet.frame_pts;
    packet.frame_adrs = reinterpret_cast<std::uintptr_t>(GetFrame(&pts));
    packet.frame_pts = pts;
    // Load DLPack Tensor
    if((reinterpret_cast<uint8_t*>(packet.frame_adrs) != nullptr) && (frame_size > 0)) {
        uint32_t width = GetWidth();
        uint32_t height = GetHeight();
        uint32_t surf_stride = GetSurfaceStride();
        uint32_t bit_depth = GetBitDepth();
        std::string type_str;
        std::vector<size_t> stride;
        if (bit_depth == 8) {
            type_str = static_cast<const char*>("|u1");
            stride.push_back(static_cast<size_t>(surf_stride));
            stride.push_back(sizeof(uint8_t));
        } else if (bit_depth <= 16) {
            type_str = static_cast<const char*>("|u2");
            stride.push_back(static_cast<size_t>(surf_stride));
            stride.push_back(sizeof(uint16_t));
        }
        // for NV12 format (also YUV444 & P016 when supported), Y always in ext_buf vector index [0]
        // The tensor shape->height will be all the Yuv planes if user specify 'FALSE' in 'SeparateYuvPlanes' argument
        float plane_height_multiplier = SeparateYuvPlanes ? 1.0 : 1.5; // 1.5 for YUV NV12
        std::vector<size_t> shape{ static_cast<size_t>(height * plane_height_multiplier), static_cast<size_t>(width)};
        packet.ext_buf[0]->LoadDLPack(shape, stride, bit_depth, type_str, (void *)packet.frame_adrs, device_id_);
        if (SeparateYuvPlanes) {
            // get surface format
            OutputSurfaceInfo* p_surf_info;
            bool ret = GetOutputSurfaceInfo(&p_surf_info);
            if (ret) {
                // for NV12 only the UV interleaved in one tensor: ext_buf vector index [1]
                if (p_surf_info->surface_format == rocDecVideoSurfaceFormat_NV12 || p_surf_info->surface_format == rocDecVideoSurfaceFormat_P016) {
                    std::vector<size_t> shape{ static_cast<size_t>(height >> 1), static_cast<size_t>(width)};
                    uintptr_t uv_offset = p_surf_info->output_pitch * p_surf_info->output_vstride; // count for possible padding
                    packet.ext_buf[1]->LoadDLPack(shape, stride, bit_depth, type_str, (void *)(packet.frame_adrs + uv_offset), device_id_);
                } else {
                    cout << "surf fmt: " << p_surf_info->surface_format << " [not supported]" << "\n";
                }
            }
        }
    }
    return py::cast(packet.frame_pts);
}

size_t PyRocVideoDecoder::CalculateRgbImageSize(OutputFormatEnum& e_output_format, OutputSurfaceInfo * p_surf_info) {
    size_t rgb_image_size = 0;
    int rgb_width = 0;
    if (p_surf_info->bit_depth == 8) {
        rgb_width = (p_surf_info->output_width + 1) & ~1; // has to be a multiple of 2 for hip colorconvert kernels
        rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * p_surf_info->output_height * 3 : rgb_width * p_surf_info->output_height * 4;
    } else {
        rgb_width = (p_surf_info->output_width + 1) & ~1;
        rgb_image_size = ((e_output_format == bgr) || (e_output_format == rgb)) ? rgb_width * p_surf_info->output_height * 3 : ((e_output_format == bgr48) || (e_output_format == rgb48)) ? rgb_width * p_surf_info->output_height * 6 : rgb_width * p_surf_info->output_height * 8;
    }
    return rgb_image_size;
}

// for python binding
py::object PyRocVideoDecoder::PyGetFrameRgb(PyPacketData& packet, int rgb_format) {
    OutputFormatEnum e_output_format = static_cast<OutputFormatEnum>(rgb_format);
    // Get YUV Frame
    int64_t pts = packet.frame_pts;
    packet.frame_adrs = reinterpret_cast<std::uintptr_t>(GetFrame(&pts));
    packet.frame_pts = pts;
    // Load DLPack Tensor
    if(reinterpret_cast<uint8_t*>(packet.frame_adrs) != nullptr) {
        // get surface info
        OutputSurfaceInfo * surf_info = nullptr;
        GetOutputSurfaceInfo(&surf_info);
        if(surf_info == nullptr)
            return py::cast(-1); // ret failure
        // get/calc new rgb image size
        size_t rgb_image_size = CalculateRgbImageSize(e_output_format, surf_info);
        if(rgb_image_size <= 0)
            return py::cast(-1); // ret failure
        // allocate 'new' RGB image device-memory if wasn't
        if(frame_ptr_rgb == nullptr) {
            HIP_API_CALL(hipMalloc((void **)&frame_ptr_rgb, rgb_image_size));
            if(frame_ptr_rgb == nullptr)
                return py::cast(-1); // ret failure
        }
        // create new instance of post process class if not created
        if(post_process_class == nullptr) {
            post_process_class = new VideoPostProcess();
        }
        // use post process instance
        VideoPostProcess * post_proc = post_process_class;
        // Get Stream, and convert YUV 2 RGB
        post_proc->ColorConvertYUV2RGB(reinterpret_cast<uint8_t*>(packet.frame_adrs), surf_info, frame_ptr_rgb, e_output_format, 0);
        // save the rgb ptr
        packet.frame_adrs_rgb = reinterpret_cast<std::uintptr_t>(frame_ptr_rgb);
        // Load DLPack Tensor
        if(reinterpret_cast<uint8_t*>(packet.frame_adrs) != nullptr) {
            uint32_t width = GetWidth();
            uint32_t height = GetHeight();
            uint32_t surf_stride = post_proc->GetRgbStride(e_output_format, surf_info);
            uint32_t bit_depth = GetBitDepth();
            std::string type_str(static_cast<const char*>("|u1"));
            std::vector<size_t> shape{ static_cast<size_t>(height), static_cast<size_t>(width), 3}; // 3 rgb channels
            std::vector<size_t> stride{ static_cast<size_t>(surf_stride), 1, 0}; // python assumes same dim for both shape & strides
            packet.ext_buf[0]->LoadDLPack(shape, stride, bit_depth, type_str, (void *)frame_ptr_rgb, device_id_);
        }
    }
    return py::cast(packet.frame_pts);
}

// for python binding
uintptr_t PyRocVideoDecoder::PyGetResizedOutputSurfaceInfo() {
    return reinterpret_cast<std::uintptr_t>(resized_surf_info);
}

// for python binding
uintptr_t PyRocVideoDecoder::PyResizeFrame(PyPacketData& packet, Dim *resized_dim, uintptr_t& in_surf_info) {
    // check params
    if(resized_dim == nullptr || in_surf_info == 0)
        return 0;
    if((reinterpret_cast<uint8_t*>(packet.frame_adrs) == nullptr) || resized_dim->w == 0 || resized_dim->h == 0)
        return 0;
    OutputSurfaceInfo *surf_info = reinterpret_cast<OutputSurfaceInfo*>(in_surf_info);
    // validate request
    if ((surf_info->output_width == resized_dim->w) && (surf_info->output_height == resized_dim->h))
        return 0;
    uint8_t *in_yuv_frame = reinterpret_cast<uint8_t*>(packet.frame_adrs);
    size_t requested_size_in_bytes = resized_dim->w * (resized_dim->h + (resized_dim->h >> 1)) * surf_info->bytes_per_pixel;
    // alloc or refill surf-info one time, and refill if size changed
    if (resized_image_size_in_bytes != requested_size_in_bytes) {
        resized_image_size_in_bytes = requested_size_in_bytes;
        if(resized_surf_info == nullptr) {
            if((resized_surf_info = reinterpret_cast<OutputSurfaceInfo*>(malloc(sizeof(OutputSurfaceInfo)))) == nullptr) {
                std::cerr << "ERROR: Failed to allocate Surface Info!" << std::endl;
                resized_image_size_in_bytes = 0;
                return 0;
            }
        }
        memcpy(resized_surf_info, surf_info, sizeof(OutputSurfaceInfo));
        resized_surf_info->output_width = resized_dim->w;
        resized_surf_info->output_height = resized_dim->h;
        resized_surf_info->output_pitch = resized_dim->w * surf_info->bytes_per_pixel;
        resized_surf_info->output_vstride = resized_dim->h;
        resized_surf_info->output_surface_size_in_bytes = resized_surf_info->output_pitch * (resized_dim->h + (resized_dim->h >> 1));

        // new size means new MEM, dealloc old one if exist
        if (frame_ptr_resized != nullptr) {
            hipError_t hip_status = hipFree(frame_ptr_resized);
            if (hip_status != hipSuccess) {
                std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
            }
            frame_ptr_resized = nullptr;
        }
    }
    // new MEM if not allocated
    if (frame_ptr_resized == nullptr)  {
        hipError_t hip_status = hipMalloc((void **)&frame_ptr_resized, resized_image_size_in_bytes);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMalloc failed to allocate the device memory for the output!" << hip_status << std::endl;
            return 0;
        }
    }
    // call resize kernel, TODO: below code assumes NV12/P016 for decoded surface. Modify to take other surface formats in future
    if (surf_info->bytes_per_pixel == 2) {
        ResizeP016(frame_ptr_resized, resized_dim->w * 2, resized_dim->w, resized_dim->h, in_yuv_frame, surf_info->output_pitch, surf_info->output_width, surf_info->output_height, (in_yuv_frame + surf_info->output_vstride * surf_info->output_pitch), nullptr, 0);
    } else {
        ResizeNv12(frame_ptr_resized, resized_dim->w, resized_dim->w, resized_dim->h, in_yuv_frame, surf_info->output_pitch, surf_info->output_width, surf_info->output_height, (in_yuv_frame + surf_info->output_vstride * surf_info->output_pitch), nullptr, 0);
    }
    // save new resized frame address
    packet.frame_adrs_resized = reinterpret_cast<std::uintptr_t>(frame_ptr_resized);
    return reinterpret_cast<std::uintptr_t>(resized_surf_info);
}

// for python binding (can not move it to header for py)
py::object PyRocVideoDecoder::PyGetNumOfFlushedFrames() {
    int32_t ret = GetNumOfFlushedFrames();
    return py::cast(ret);
}

// for python binding
py::object PyRocVideoDecoder::PyReleaseFrame(PyPacketData& packet) {
    bool ret = ReleaseFrame(packet.frame_pts);
    return py::cast(ret);
}

// for python binding
py::object PyRocVideoDecoder::PySaveFrameToFile(std::string& output_file_name_in, uintptr_t& surf_mem, uintptr_t& surface_info, OutputFormatEnum e_output_format) {
    std::string output_file_name = output_file_name_in.c_str();
    OutputSurfaceInfo *p_surf_info;
    bool ret = true;
    if (surface_info)
        p_surf_info = reinterpret_cast<OutputSurfaceInfo*>(surface_info);
    else
        ret = GetOutputSurfaceInfo(&p_surf_info);
    if(surf_mem && ret) {
        size_t image_size = 0; // 0 size == rgb frame
        if (e_output_format != OutputFormatEnum::native) { // native == YUV frame
            image_size = CalculateRgbImageSize(e_output_format, p_surf_info);
        }
        SaveFrameToFile(output_file_name, (void *)surf_mem, p_surf_info, image_size);
    }
    return py::cast<py::none>(Py_None);
}

// for python binding
std::shared_ptr<ConfigInfo> PyRocVideoDecoder::PyGetDeviceinfo() {
    GetDeviceinfo(configInfo.get()->device_name, configInfo.get()->gcn_arch_name, configInfo.get()->pci_bus_id, configInfo.get()->pci_domain_id, configInfo.get()->pci_device_id);
    return configInfo; 
}

// for python binding
uintptr_t PyRocVideoDecoder::PyGetOutputSurfaceInfo() {
    OutputSurfaceInfo *l_surface_info;
    bool ret = GetOutputSurfaceInfo(&l_surface_info);
    if (ret) {
       return reinterpret_cast<std::uintptr_t>(l_surface_info);
    }
    return 0;
}

// for python binding
py::int_ PyRocVideoDecoder::PyGetWidth() {
    return py::int_(static_cast<int>(GetWidth()));
}

// for python binding
py::int_ PyRocVideoDecoder::PyGetHeight() {
    return py::int_(static_cast<int>(GetHeight()));
}

// for python binding
py::int_ PyRocVideoDecoder::PyGetFrameSize() {
    return py::int_(static_cast<int>(GetFrameSize()));
}

// for python binding
py::int_ PyRocVideoDecoder::PyGetStride() {
    return py::int_(static_cast<int>(GetSurfaceStride()));
}

// for python binding
py::object PyRocVideoDecoder::PyCodecSupported(int device_id, rocDecVideoCodec codec_id, uint32_t bit_depth) {
    bool ret = CodecSupported(device_id, codec_id, bit_depth);
    return py::cast(ret);
}

uint32_t PyRocVideoDecoder::PyGetBitDepth() {
    return GetBitDepth();
}

#if ROCDECODE_CHECK_VERSION(0,6,0)
// for python binding, Session overhead refers to decoder initialization and deinitialization time
py::object PyRocVideoDecoder::PyAddDecoderSessionOverHead(int session_id, double duration) {
    AddDecoderSessionOverHead(static_cast<std::thread::id>(session_id), duration);
    return py::cast<py::none>(Py_None);
}

// for python binding, Session overhead refers to decoder initialization and deinitialization time
py::object PyRocVideoDecoder::PyGetDecoderSessionOverHead(int session_id) {
    return py::cast(GetDecoderSessionOverHead(static_cast<std::thread::id>(session_id)));
}

#endif

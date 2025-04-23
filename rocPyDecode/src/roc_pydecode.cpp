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

using namespace std;

PYBIND11_MODULE(rocpydecode, m) {
 
    m.doc() = "Python bindings for the C++ portions of rocDecode ..";

    // convert betweeen demuxer & decoder
    m.def("AVCodec2RocDecVideoCodec", &ConvertAVCodec2RocDecVideoCodec, "Convert AVCodecID to rocDecVideoCodec ID");
    m.def("AVCodecString2RocDecVideoCodec", &ConvertAVCodecString2RocDecVideoCodec, "Convert AVCodec string to rocDecVideoCodec ID");
    
    m.def("GetRocPyDecPacket", [](int pts, int size, py::buffer buffer) {
        std::shared_ptr<PyPacketData> packet = make_shared<PyPacketData>();
        packet->frame_pts = static_cast<int64_t>(pts);
        packet->bitstream_size = static_cast<int64_t>(size);
        // process py::buffer object to an address ptr for bitstream
        py::buffer_info buffer_info = buffer.request();
        packet->bitstream_adrs = reinterpret_cast<uintptr_t>(buffer_info.ptr);
        return packet;
    }, "Convert packet info from user to rocpydecode's PyPacketData");



    // ------
    // Types:
    // ------
    py::module types_m = m.def_submodule("decTypes");
    types_m.doc() = "Datatypes and options used by rocDecode";            

    // current version
    // Todo: to be changed to match version on CMakeLists with every future version update
    m.attr("__version__") = py::str("0.4.0");

    // rocDecVideoSurfaceFormat
    py::enum_<rocDecVideoSurfaceFormat>(types_m, "rocDecVideoSurfaceFormat")
        .value("rocDecVideoSurfaceFormat_NV12",rocDecVideoSurfaceFormat_NV12)					
        .value("rocDecVideoSurfaceFormat_P016",rocDecVideoSurfaceFormat_P016)					
        .value("rocDecVideoSurfaceFormat_YUV444",rocDecVideoSurfaceFormat_YUV444)				
        .value("rocDecVideoSurfaceFormat_YUV444_16Bit",rocDecVideoSurfaceFormat_YUV444_16Bit) 	
		.export_values(); 
                
    py::enum_<RocdecVideoPacketFlags>(types_m,"RocdecVideoPacketFlags","Video Packet Flags")
        .value("ROCDEC_PKT_ENDOFSTREAM",ROCDEC_PKT_ENDOFSTREAM)
        .value("ROCDEC_PKT_TIMESTAMP",ROCDEC_PKT_TIMESTAMP)
        .value("ROCDEC_PKT_DISCONTINUITY",ROCDEC_PKT_DISCONTINUITY)
        .value("ROCDEC_PKT_ENDOFPICTURE",ROCDEC_PKT_ENDOFPICTURE)
        .value("ROCDEC_PKT_NOTIFY_EOS",ROCDEC_PKT_NOTIFY_EOS)
        .export_values(); 

    py::enum_<rocDecVideoCodec>(types_m,"rocDecVideoCodec","Video Codec") 
        .value("rocDecVideoCodec_AVC",rocDecVideoCodec_AVC)            
        .value("rocDecVideoCodec_HEVC",rocDecVideoCodec_HEVC)          
        .export_values(); 

    py::enum_<OutputFormatEnum>(types_m,"OutputFormatEnum","Types of images")
        .value("native",native)
        .value("bgr",bgr)
        .value("bgr48",bgr48)
        .value("rgb",rgb)
        .value("rgb48",rgb48)
        .value("bgra",bgra)
        .value("bgra64",bgra64)
        .value("rgba",rgba)
        .value("rgba64",rgba64)
       .export_values();

    // ---------
    // PyExport
    // ---------
    PyExportInitializer(m);

    // -----------------------------
    // User Demuxer 'PyVideoDemuxer'
    // -----------------------------
    PyVideoDemuxerInitializer(m);

    // ------------------------------------------------
    // StreamProvider 'PyVideoStreamProvider' for demux
    // ------------------------------------------------
    PyVideoStreamProviderInitializer(m);

    // --------------------------------------
    // AMD Video Decoder 'PyRocVideoDecoder'
    // --------------------------------------
    PyRocVideoDecoderInitializer(m);

    // --------------------------------------
    // AMD Video Decoder 'PyRocVideoDecoderCpu' -- FFMpeg Decode
    // --------------------------------------
    PyRocVideoDecoderCpuInitializer(m);

    // ----------------
    // Structures:
    // ----------------

    // OutputSurfaceInfo
    py::class_<OutputSurfaceInfo>(m, "OutputSurfaceInfo")
        .def(py::init<>())
        .def_readwrite("output_width",&OutputSurfaceInfo::output_width)	            	            	  
        .def_readwrite("output_height",&OutputSurfaceInfo::output_height)		            	          
        .def_readwrite("output_pitch",&OutputSurfaceInfo::output_pitch)	     	            	                             
        .def_readwrite("output_vstride",&OutputSurfaceInfo::output_vstride)	   	            	          
        .def_readwrite("bytes_per_pixel",&OutputSurfaceInfo::bytes_per_pixel)		            	      
        .def_readwrite("bit_depth",&OutputSurfaceInfo::bit_depth)	           	            	          
        .def_readwrite("num_chroma_planes",&OutputSurfaceInfo::num_chroma_planes)	            	      
        .def_readwrite("output_surface_size_in_bytes",&OutputSurfaceInfo::output_surface_size_in_bytes)   
        .def_readwrite("surface_format",&OutputSurfaceInfo::surface_format)			            	      
        .def_readwrite("mem_type",&OutputSurfaceInfo::mem_type);                                                  				
  
    // Rect
    py::class_<Rect>(m, "Rect")
        .def(py::init<>())
        .def_readwrite("left",&Rect::left)
        .def_readwrite("top",&Rect::top)
        .def_readwrite("right",&Rect::right)
        .def_readwrite("bottom",&Rect::bottom);

    // Dim
    py::class_<Dim>(m, "Dim")
        .def(py::init<>())
        .def_readwrite("width",&Dim::w)
        .def_readwrite("height",&Dim::h);

    // PyPacketData
    py::class_<PyPacketData, shared_ptr<PyPacketData>>(m, "PyPacketData", py::module_local())
        .def(py::init<>())
        .def_readwrite("end_of_stream", &PyPacketData::end_of_stream)
        .def_readwrite("pkt_flags",     &PyPacketData::pkt_flags)
        .def_readwrite("frame_pts",     &PyPacketData::frame_pts)
        .def_readwrite("frame_size",    &PyPacketData::frame_size)
        .def_readwrite("frame_adrs",    &PyPacketData::frame_adrs)
        .def_readwrite("bitstream_size",    &PyPacketData::bitstream_size)
        .def_readwrite("bitstream_adrs",    &PyPacketData::bitstream_adrs)
        .def_readwrite("frame_adrs_rgb", &PyPacketData::frame_adrs_rgb)
        .def_readwrite("frame_adrs_resized", &PyPacketData::frame_adrs_resized)
        .def_readwrite("ext_buf",        &PyPacketData::ext_buf)

        // DL Pack Tensor
        .def_property_readonly("shapeY", [](std::shared_ptr<PyPacketData>& self) {
            return self->ext_buf[0]->shape();
            }, "Get the shape of the Y plane buffer as an array")
        .def_property_readonly("shapeUV", [](std::shared_ptr<PyPacketData>& self) {
            return self->ext_buf[1]->shape();
            }, "Get the shape of the U plane buffer as an array")
        .def_property_readonly("shapeU", [](std::shared_ptr<PyPacketData>& self) {
            return self->ext_buf[1]->shape();
            }, "Get the shape of the U plane buffer as an array")
        .def_property_readonly("shapeV", [](std::shared_ptr<PyPacketData>& self) {
            return self->ext_buf[2]->shape();
            }, "Get the shape of the V plane buffer as an array")
        .def_property_readonly("shape", [](std::shared_ptr<PyPacketData>& self) {
            return self->ext_buf[0]->shape();
            }, "Get the shape of the buffer as an array")
        .def_property_readonly("strides", [](std::shared_ptr<PyPacketData>& self) {
                return self->ext_buf[0]->strides();
            }, "Get the strides of the buffer")
        .def_property_readonly("dtype", [](std::shared_ptr<PyPacketData>& self) {
                return self->ext_buf[0]->dtype();
            }, "Get the data type of the buffer")
        .def("__dlpack__", [](std::shared_ptr<PyPacketData>& self, py::object stream) {
            return self->ext_buf[0]->dlpack(stream);
            }, py::arg("stream") = NULL, "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", [](std::shared_ptr<PyPacketData>& self) {
                return py::make_tuple(py::int_(static_cast<int>(DLDeviceType::kDLROCM)),
                        py::int_(static_cast<int>(0)));
            }, "Get the device associated with the buffer");   

    // ConfigInfo
    py::class_<ConfigInfo, shared_ptr<ConfigInfo>>(m, "ConfigInfo", py::module_local())
        .def(py::init<>())
        .def_readwrite("device_name",   &ConfigInfo::device_name)
        .def_readwrite("gcn_arch_name", &ConfigInfo::gcn_arch_name)
        .def_readwrite("pci_bus_id",    &ConfigInfo::pci_bus_id)
        .def_readwrite("pci_domain_id", &ConfigInfo::pci_domain_id)
        .def_readwrite("pci_device_id", &ConfigInfo::pci_device_id);
}
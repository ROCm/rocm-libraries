# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import rocpydecode as rocpydec
import rocpydecode.decTypes as dectypes
import numpy as np


def GetOutputFormat(rgb_format) -> dectypes.OutputFormatEnum:
    out_format = dectypes.OutputFormatEnum(rgb_format)
    return out_format

def GetRocDecCodecID(codec_id) -> dectypes.rocDecVideoCodec:
    rocCodecId = None
    if isinstance(codec_id, int):
        rocCodecId = rocpydec.AVCodec2RocDecVideoCodec(codec_id)
    if isinstance(codec_id, str):
        rocCodecId = rocpydec.AVCodecString2RocDecVideoCodec(codec_id)
    return rocCodecId

def GetRectangle(crop_rect: dict) -> rocpydec.Rect:
    p_crop_rect = rocpydec.Rect()
    if (crop_rect is not None):
        p_crop_rect.left = crop_rect[0]
        p_crop_rect.top = crop_rect[1]
        p_crop_rect.right = crop_rect[2]
        p_crop_rect.bottom = crop_rect[3]
    return p_crop_rect

def GetDim(p_dim_wd: dict) -> rocpydec.Dim:
    dim_wd = rocpydec.Dim()
    if (p_dim_wd is not None):
        dim_wd.width = p_dim_wd[0]
        dim_wd.height = p_dim_wd[1]
    return dim_wd

def GetOutputSurfaceInfo():
    surf_info_struct = rocpydec.OutputSurfaceInfo()
    return surf_info_struct

def GetRocPyDecPacket(pts, size, buffer):
    #pts_us = int(pts * 1000 * 1000)  #TBD: if needed in microseconds
    return rocpydec.GetRocPyDecPacket(0 if pts == None else int(pts), size, buffer)

class decoder(object):
    def __init__(
            self,
            codec,
            device_id = 0,
            mem_type = 0,
            b_force_zero_latency = False,
            crop_rect = None,
            max_width = 0,
            max_height = 0,
            clk_rate = 1000):
        p_crop_rect = GetRectangle(crop_rect)
        self.viddec = rocpydec.PyRocVideoDecoder(
            device_id,
            mem_type,
            codec,
            b_force_zero_latency,
            p_crop_rect,
            max_width,
            max_height,
            clk_rate)

    def GetGpuInfo(self):
        return self.viddec.GetDeviceinfo()

    def DecodeFrame(self, packet) -> int:
        return self.viddec.DecodeFrame(packet)

    def GetFrameYuv(self, packet, separate_planes = False):
        pts = self.viddec.GetFrameYuv(packet, separate_planes)
        return pts

    def GetFrameRgb(self, packet, rgb_format):
        pts = self.viddec.GetFrameRgb(packet, rgb_format)
        return pts

    def ResizeFrame(self, packet, resize_dim, surface_info):
        resize_dim = GetDim(resize_dim)
        return self.viddec.ResizeFrame(packet, resize_dim, surface_info)

    def GetWidth(self) -> int:
        return self.viddec.GetWidth()

    def GetHeight(self) -> int:
        return self.viddec.GetHeight()

    def GetStride(self) -> int:
        return self.viddec.GetStride()

    def GetFrameSize(self) -> int:
        return self.viddec.GetFrameSize()

    def GetOutputSurfaceInfo(self):
        return self.viddec.GetOutputSurfaceInfo()

    def GetResizedOutputSurfaceInfo(self):
        return self.viddec.GetResizedOutputSurfaceInfo()

    def SaveFrameToFile(self, output_file_path, frame_adrs, surface_info = 0, output_format:dectypes.OutputFormatEnum = dectypes.OutputFormatEnum.native):
        return self.viddec.SaveFrameToFile( output_file_path, frame_adrs, surface_info, output_format)

    def ReleaseFrame(self, packet):
        self.viddec.ReleaseFrame(packet)
        return

    def GetNumOfFlushedFrames(self):
        return self.viddec.GetNumOfFlushedFrames()

    def SetReconfigParams(self, flush_mode, out_file_name):
        return self.viddec.SetReconfigParams(flush_mode, out_file_name)

    def AddDecoderSessionOverHead(self, session_id, duration):
        if(hasattr(self.viddec,"AddDecoderSessionOverHead")):
            return self.viddec.AddDecoderSessionOverHead(session_id, duration)
        else:
            return None

    def GetDecoderSessionOverHead(self, session_id):
        if(hasattr(self.viddec, "GetDecoderSessionOverHead")):
            return self.viddec.GetDecoderSessionOverHead(session_id)
        else:
            return None

    def IsCodecSupported(self, device_id, codec_id, bit_depth):
        return self.viddec.IsCodecSupported(device_id, codec_id, bit_depth)
    
    def GetBitDepth(self):
        return self.viddec.GetBitDepth()

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

class stream_provider(object):
    def __init__(self, input_file_path: str):
        self.stream_provider = rocpydec.PyFileStreamProvider(input_file_path)

    def GetFileStreamProvider(self):
         return self.stream_provider

class demuxer(object):
    def __init__(self, name):
        if isinstance(name, str):
            self.vidmux = rocpydec.PyVideoDemuxer(name)
        elif isinstance(name, stream_provider):
            self.vidmux = rocpydec.PyVideoDemuxer(name.GetFileStreamProvider())

    def GetCodecId(self)-> int:
        return self.vidmux.GetCodecId() 

    def DemuxFrame(self):
         return self.vidmux.DemuxFrame()

    def SeekFrame(self, frame_number, seek_mode, seek_criteria):
         return self.vidmux.SeekFrame(frame_number, seek_mode, seek_criteria)

    def GetBitDepth(self):
        return self.vidmux.GetBitDepth()


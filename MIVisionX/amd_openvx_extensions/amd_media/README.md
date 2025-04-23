# OpenVX Media Extension Library

`vx_amd_media` is an OpenVX AMD media extension module. This module has mainly two OpenVX extension nodes. `com.amd.amd_media.decode` node for video/jpeg decoding and `com.amd.amd_media.encode` node for video encoding

### List of OpenVX Media Extension Nodes:

| Node name                | Function           | Parameters                                                     |
|--------------------------|--------------------|----------------------------------------------------------------|
| com.amd.amd_media.decode | Video/JPEG decoder | Supports MP4/JPEG input files and outputs YUV or RGB           |
| com.amd.amd_media.encode | Video encoder      | Supports input YUV/RGB input and .264 elementary stream output |

## Build Instructions

### Pre-requisites

* AMD OpenVX&trade; library
* FFMPEG 4.0 or above, installed via MIVisionX setup script or download from following link
    - [download](https://ffmpeg.org/download.html)
* amdgpu Linux mesa driver for hardware support. Install with --no-dkms after installing ROCm.
    - [amdgpu](https://amdgpu-install.readthedocs.io/en/latest/)
    
### Example 1: decode video with runvx using software decoder

Following is an example gdf to decode 1 video stream using CPU decoder and decoded images will be written to output.yuv file

``` 
import vx_amd_media

# read input sequences
data vid1 = scalar:STRING,"1,<fname_with_full_path.mp4>:0"
data nvimg  = image:1920,1080,NV12:write,output.yuv
data loop = scalar:INT32,0
data opencl_out = scalar:INT32,0
node com.amd.amd_media.decode vid1 nvimg NULL loop opencl_out
```

### Example 2: decode video with runvx using hardware decoder

Following is an example gdf to encode 1 video stream using hardware and decoded images will be written to output.yuv file

``` 
import vx_amd_media

# read input sequences
data vid1 = scalar:STRING,"1,<fname_with_full_path.mp4>:1"
data nvimg  = image:1920,1080,NV12:write,output.yuv
data loop = scalar:INT32,0
data opencl_out = scalar:INT32,0
node com.amd.amd_media.decode vid1 nvimg NULL loop opencl_out
```

### Example 3: Encoding from yuv image to .264 file

Following is an example gdf to encode to .h264 stream from a YUV input file

Sample command: runvx -frames:<#framestoencode> file <encoder.gdf>

``` 
import vx_amd_media

# read input sequences
data yuvimg  = image:1920,1080,NV12:read,input.yuv
data vid1 = scalar:STRING,"fname_with_full_path.264"
data aux_output = array:UINT8,256
data gpu_mode = scalar:BOOL,FALSE
node com.amd.amd_media.encode vid1 yuvimg NULL aux_output gpu_mode
```

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.

# rocJPEG changelog

Documentation for rocJPEG is available at
[https://rocm.docs.amd.com/projects/rocJPEG/en/latest/](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/)

## rocJPEG 0.9.0 (Unreleased)

### Resolved issues

* Fixed a bug that prevented copying the decoded image into the output buffer when the output buffer is larger than the input image.

## rocJPEG 0.8.0 for ROCm 6.4

### Changed

* AMD Clang++ is now the default CXX compiler.
* The jpegDecodeMultiThreads sample has been renamed to jpegDecodePerf, and batch decoding has been added to this sample instead of single image decoding for improved performance.

## rocJPEG 0.6.0 for ROCm 6.3.0

### Changes

* Supported initial enablement of the rocJPEG library
* Supported JPEG chroma subsampling:
  * YUV 4:4:4
  * YUV 4:4:0
  * YUV 4:2:2
  * YUV 4:2:0
  * YUV 4:0:0
* Supported various output image format:
  * Native (i.e., native unchanged output from VCN Hardware, it can be either packed YUV or planar YUV)​
  * YUV planar​
  * Y only​
  * RGB​
  * RGB planar
* Supported single-image and batch-image decoding​
* Supported Different partition modes on MI300 series​
* Supported region-of-interest (ROI) decoding​
* Supported color space conversion from YUV to RGB
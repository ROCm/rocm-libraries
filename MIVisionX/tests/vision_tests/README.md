# OpenVX Vision Function Tests

## Script to run vision tests

```
python runVisionTests.py --help
```


usage:

```
runVisionTests.py [--runvx_directory RUNVX_DIRECTORY] -- required
                  [--hardware_mode HARDWARE_MODE]
                  [--list_tests LIST_TESTS]
                  [--test_filter TEST_FILTER]
                  [--num_frames NUM_FRAMES]
                  [--functionality FUNCTIONALITY]
                  [--backend_type BACKEND_TYPE]
                  [--profiling PROFILING]

Arguments:
  -h, --help        show this help message and exit
  --runvx_directory RunVX Executable Directory - required
  --hardware_mode   OpenVX Vision Function Target - optional (default:CPU [options:CPU/GPU])
  --list_tests      List Vision Performance Tests - optional (default:no [options:no/yes])
  --test_filter     Vision Performance Test Filter - optional (default:0 [range:1 - N])
  --num_frames      Run Test for X number of frames - optional (default:1000 [range:1 - N])
  --functionality   Vision Functionality Tests Enabled - optional (default:yes [options:no/yes])
  --backend_type    OpenVX Backend type - optional (default:CPU [options:CPU/HIP/OCL])
  --profiling       Enable GPU profiling with ROCm profiler (rocprof) - optional (default:no [options:yes/no])
```

## Vision Functionality Test GDFs

```
01_absDiff
02_accumulate
03_accumulateSquared
04_accumulateWeighted
05_add
06_and
07_box
08_canny
09_channelCombine
10_channelExtract
11_colorConvert
12_convertDepth
13_convolve
14_dilate
15_equalizeHistogram
16_erode
17_fastCorners
18_gaussian
19_harrisCorners
20_halfScaleGaussian
21_histogram
22_integralImage
23_magnitude
24_meanStdDev
25_median
26_minMaxLoc
27_multiply
28_not
29_opticalFlowLK
30_or
31_phase
32_gaussianPyramid
33_remap
34_scaleImage
35_sobel
36_subtract
37_tableLookup
38_threshold
39_warpAffine
40_warpPerspective
41_xor
```

## Vision Performance Tests

```
Test ID - Test Name
1   AbsDiff_S16_S16S16_Sat
2   Add_U8_U8U8_Wrap
3   Add_U8_U8U8_Sat
4   Add_S16_U8U8_Wrap
5   Add_S16_S16U8_Wrap
6   Add_S16_S16U8_Sat
7   Add_S16_S16S16_Wrap
8   Add_S16_S16S16_Sat
9   Sub_U8_U8U8_Wrap
10  Sub_U8_U8U8_Sat
11  Sub_S16_U8U8_Wrap
12  Sub_S16_S16U8_Wrap
13  Sub_S16_S16U8_Sat
14  Sub_S16_U8S16_Wrap
15  Sub_S16_U8S16_Sat
16  Sub_S16_S16S16_Wrap
17  Sub_S16_S16S16_Sat
18  Mul_U8_U8U8_Wrap_Trunc
19  Mul_U8_U8U8_Wrap_Round
20  Mul_U8_U8U8_Sat_Trunc
21  Mul_U8_U8U8_Sat_Round
22  Mul_S16_U8U8_Wrap_Trunc
23  Mul_S16_U8U8_Wrap_Round
24  Mul_S16_U8U8_Sat_Trunc
25  Mul_S16_U8U8_Sat_Round
26  Mul_S16_S16U8_Wrap_Trunc
27  Mul_S16_S16U8_Wrap_Round
28  Mul_S16_S16U8_Sat_Trunc
29  Mul_S16_S16U8_Sat_Round
30  Mul_S16_S16S16_Wrap_Trunc
31  Mul_S16_S16S16_Wrap_Round
32  Mul_S16_S16S16_Sat_Trunc
33  Mul_S16_S16S16_Sat_Round
34  Magnitude_S16_S16S16
35  Phase_U8_S16S16
36  WeightedAverage_U8_U8U8
37  And_U8_U8U8
38  Or_U8_U8U8
39  Xor_U8_U8U8
40  Not_U8_U8
41  Lut_U8_U8
42  ColorDepth_U8_S16_Wrap
43  ColorDepth_U8_S16_Sat
44  ColorDepth_S16_U8
45  ChannelExtract_U8_U24_Pos0
46  ChannelExtract_U8_U24_Pos1
47  ChannelExtract_U8_U24_Pos2
48  ChannelExtract_U8_U32_Pos0_UYVY
49  ChannelExtract_U8_U32_Pos1_YUYV
50  ChannelExtract_U8_U32_Pos2_UYVY
51  ChannelExtract_U8_U32_Pos3_YUYV
52  ChannelExtract_U8_U32_Pos0_RGBX
53  ChannelExtract_U8_U32_Pos1_RGBX
54  ChannelExtract_U8_U32_Pos2_RGBX
55  ChannelExtract_U8_U32_Pos3_RGBX
56  ChannelExtract_U8U8U8_U24
57  ChannelExtract_U8U8U8_U32
58  ChannelExtract_U8U8U8U8_U32
59  ChannelCombine_U32_U8U8U8U8_RGBX
60  ColorConvert_RGB_RGBX
61  ColorConvert_RGB_UYVY
62  ColorConvert_RGB_YUYV
63  ColorConvert_RGB_IYUV
64  ColorConvert_RGB_NV12
65  ColorConvert_RGB_NV21
66  ColorConvert_RGBX_RGB
67  ColorConvert_RGBX_UYVY
68  ColorConvert_RGBX_YUYV
69  ColorConvert_RGBX_IYUV
70  ColorConvert_RGBX_NV12
71  ColorConvert_RGBX_NV21
72  ColorConvert_IYUV_RGB
73  ColorConvert_IYUV_RGBX
74  FormatConvert_IYUV_UYVY
75  FormatConvert_IYUV_YUYV
76  ColorConvert_NV12_RGB
77  ColorConvert_NV12_RGBX
78  FormatConvert_NV12_UYVY
79  FormatConvert_NV12_YUYV
80  ColorConvert_YUV4_RGB
81  ColorConvert_YUV4_RGBX
82  Box_U8_U8_3x3
83  Dilate_U8_U8_3x3
84  Erode_U8_U8_3x3
85  Median_U8_U8_3x3
86  Gaussian_U8_U8_3x3
87  ScaleGaussianHalf_U8_U8_3x3
88  ScaleGaussianHalf_U8_U8_5x5
89  Convolve_U8_U8_3x3
90  Convolve_S16_U8_3x3
91  Sobel_S16S16_U8_3x3_GXY
92  Sobel_S16_U8_3x3_GX
93  Threshold_U8_U8_Binary
94  Threshold_U8_U8_Range
95  Threshold_U8_S16_Binary
96  Threshold_U8_S16_Range
97  ScaleImage_U8_U8_Nearest
98  ScaleImage_U8_U8_Bilinear
99  ScaleImage_U8_U8_Bilinear_Replicate
100 ScaleImage_U8_U8_Bilinear_Constant
101 ScaleImage_U8_U8_Area
102 WarpAffine_U8_U8_Nearest
103 WarpAffine_U8_U8_Nearest_Constant
104 WarpAffine_U8_U8_Bilinear
105 WarpAffine_U8_U8_Bilinear_Constant
```

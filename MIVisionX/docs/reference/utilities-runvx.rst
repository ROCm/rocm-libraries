.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _runvx-ref:

******************************************
RunVX Reference
******************************************

RunVX is a command-line tool to execute OpenVX graphs, with a simple, easy-to-use interface. It encapsulates most of the routine OpenVX calls, thus speeding up development and enabling rapid prototyping. As input, RunVX takes a graph description format (GDF) file. The GDF file provides a simple and intuitive syntax to describe the various data, nodes, and their dependencies. The ``runvx`` tool also provides file read/write, data compares, image and keypoint data visualization.

.. note:: 
    ``runvx`` uses OpenCV when it is available for camera capture and image display.

Prerequisites
=============

* **CPU**: AMD64
* **GPU**: AMD Radeon Graphics [optional]
* Windows: install the latest drivers and OpenCL SDK download [optional]
* Linux: install ROCm [optional]
* Install ``libssl-dev`` on Linux [optional]
* OpenCV `3.4+` [optional]

  + for Windows set `OpenCV_DIR` environment variable to point to the `OpenCV/build` folder


RunVX Usage and GDF Syntax
==========================

.. code-block:: shell

    runvx.exe [options] <file.gdf> [argument(s)]
    runvx.exe [options] node <kernelName> [argument(s)]
    runvx.exe [options] shell [argument(s)]
        
The argument(s) are data objects created using ``<data-description>`` syntax.
The arguments can be accessed from inside GDF as ``$1``, ``$2``, etc.

The available command-line options are:

.. code-block:: shell

    -h
        Show full help.

    -v
        Turn on verbose logs.

    -root:<directory>
        Replace ~ in filenames with <directory> in the command-line and
        GDF file. The default value of '~' is current working directory.

    -frames:[<start>:]<end>|eof|live
        Run the graph/node for specified frames or until eof or just as live.
        Use live to indicate that input is live until aborted by user.

    -affinity:CPU|GPU[<device-index>]
        Set context affinity to CPU or GPU.

    -dump-profile
        Print performance profiling information after graph launch.

    -discard-compare-errors
        Continue graph processing even if compare mismatches occur.

    -disable-virtual
        Replace all virtual data types in GDF with non-virtual data types.
        Use of this flag (i.e. for debugging) can make a graph run slower.

    -dump-data-config:<dumpFilePrefix>,<object-type>[,object-type[...]]
        Automatically dump all non-virtual objects of specified object types
        into files '<dumpFilePrefix>dumpdata_####_<object-type>_<object-name>.raw'

    -discard-commands:<cmd>[,cmd[...]]
        Discard the listed commands.
    

The supported list of OpenVX built-in kernel names is given below:

* ``org.khronos.openvx.color_convert``
* ``org.khronos.openvx.channel_extract``
* ``org.khronos.openvx.channel_combine``
* ``org.khronos.openvx.sobel_3x3``
* ``org.khronos.openvx.magnitude``
* ``org.khronos.openvx.phase``
* ``org.khronos.openvx.scale_image``
* ``org.khronos.openvx.table_lookup``
* ``org.khronos.openvx.histogram``
* ``org.khronos.openvx.equalize_histogram``
* ``org.khronos.openvx.absdiff``
* ``org.khronos.openvx.mean_stddev``
* ``org.khronos.openvx.threshold``
* ``org.khronos.openvx.integral_image``
* ``org.khronos.openvx.dilate_3x3``
* ``org.khronos.openvx.erode_3x3``
* ``org.khronos.openvx.median_3x3``
* ``org.khronos.openvx.box_3x3``
* ``org.khronos.openvx.gaussian_3x3``
* ``org.khronos.openvx.custom_convolution``
* ``org.khronos.openvx.gaussian_pyramid``
* ``org.khronos.openvx.accumulate``
* ``org.khronos.openvx.accumulate_weighted``
* ``org.khronos.openvx.accumulate_square``
* ``org.khronos.openvx.minmaxloc``
* ``org.khronos.openvx.convertdepth``
* ``org.khronos.openvx.canny_edge_detector``
* ``org.khronos.openvx.and``
* ``org.khronos.openvx.or``
* ``org.khronos.openvx.xor``
* ``org.khronos.openvx.not``
* ``org.khronos.openvx.multiply``
* ``org.khronos.openvx.add``
* ``org.khronos.openvx.subtract``
* ``org.khronos.openvx.warp_affine``
* ``org.khronos.openvx.warp_perspective``
* ``org.khronos.openvx.harris_corners``
* ``org.khronos.openvx.fast_corners``
* ``org.khronos.openvx.optical_flow_pyr_lk``
* ``org.khronos.openvx.remap``
* ``org.khronos.openvx.halfscale_gaussian``
        
The supported list of AMD's built-in kernel names is given below:

* ``com.amd.nn_extension.argmax_layer``
* ``com.amd.nn_extension.batch_normalization_layer``
* ``com.amd.nn_extension.cast_layer``
* ``com.amd.nn_extension.concat_layer``
* ``com.amd.nn_extension.crop_layer``
* ``com.amd.nn_extension.crop_and_resize_layer``
* ``com.amd.nn_extension.detection_output``
* ``com.amd.nn_extension.permute_layer``
* ``com.amd.nn_extension.prior_box_layer``
* ``com.amd.nn_extension.scale_layer``
* ``com.amd.nn_extension.slice_layer``
* ``com.amd.nn_extension.convert_image_to_tensor``
* ``com.amd.nn_extension.convert_tensor_to_image``
* ``com.amd.nn_extension.tensor_exp``
* ``com.amd.nn_extension.tensor_log``
* ``com.amd.nn_extension.upsample_nearest_layer``
* ``com.amd.nn_extension.reshape_layer``
* ``com.amd.nn_extension.gather_layer``
* ``com.amd.nn_extension.topk_layer``
* ``com.amd.nn_extension.nms_layer``


The available GDF commands are:

.. code-block:: shell

    import <libraryName>
        Import kernels in a library using vxLoadKernel API.

    type <typeName> userstruct:<size-in-bytes>
        Create an OpenVX user defined structure using vxRegisterUserStruct API.
        The <typeName> can be used as a type in array object.

    data <dataName> = <data-description>
        Create an OpenVX data object in context using the below syntax for
        <data-description>:
            array:<data-type>,<capacity>
            convolution:<columns>,<rows>
            distribution:<numBins>,<offset>,<range>
            delay:<exemplar>,<slots>
            image:<width>,<height>,<image-format>[,<range>][,<space>]
            uniform-image:<width>,<height>,<image-format>,<uniform-pixel-value>
            image-from-roi:<master-image>,rect{<start-x>;<start-y>;<end-x>;<end-y>}
            image-from-handle:<image-format>,{<dim-x>;<dim-y>;<stride-x>;<stride-y>}[+...],<memory-type>
            image-from-channel:<master-image>,<channel>
            lut:<data-type>,<count>
            matrix:<data-type>,<columns>,<rows>
            pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<image-format>
            remap:<srcWidth>,<srcHeight>,<dstWidth>,<dstHeight>
            scalar:<data-type>,<value>
            threshold:<thresh-type>,<input-data-type><output-data-type>
            tensor:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>
            tensor-from-roi:<master-tensor>,<num-of-dims>,{<start0>,<start1>,...},{<end0>,<end1>,...}
            tensor-from-handle:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>,{<stride0>,<stride1>,...},<num-alloc-handles>,<memory-type>
				
          For virtual object in default graph use the below syntax for
          <data-description>:
              virtual-array:<data-type>,<capacity>
              virtual-image:<width>,<height>,<image-format>
              virtual-pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<image-format>
              virtual-tensor:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>

          where:
              <master-image> can be name of a image data object (including $1, $2, ...)
              <master-tensor> can be name of a tensor data object (including $1, $2, ...)
              <exemplar> can be name of a data object (including $1, $2, ...)
              <thresh-type> can be BINARY,RANGE
              <uniform-pixel-value> can be an integer or {<byte>;<byte>;...}
              <image-format> can be RGB2,RGBX,IYUV,NV12,U008,S016,U001,F032,...
              <data-type> can be UINT8,INT16,INT32,UINT32,FLOAT32,ENUM,BOOL,SIZE,
                                 KEYPOINT,COORDINATES2D,RECTANGLE,<typeName>,...
              <range> can be vx_channel_range_e enums FULL or RESTRICTED
              <space> can be vx_color_space_e enums BT709 or BT601_525 or BT601_625

    node <kernelName> [<argument(s)>]
        Create a node of specified kernel in the default graph with specified node arguments. 
        Node arguments have to be OpenVX data objects created earlier in GDF or data objects
        specified on command-line accessible as $1, $2, etc. For scalar enumerations as node
        arguments, use !<enumName> syntax (e.g., !VX_CHANNEL_Y for channel_extract node).
        For optional arguments, use NULL/null to set the node argument as a null value.
        (e.g. node com.amd.nn_extension.slice_layer input output starts ends NULL steps)

    include <file.gdf>
        Specify inclusion of another GDF file.

    shell
        Start a shell command session.

    set <option> [<value>]
        Specify or query the following global options:
            set verbose [on|off]
                Turn on/off verbose option.
            set frames [[<start-frame>:]<end-frame>|eof|live|default]
                Specify input frames to be processed. Here are some examples:
                    set frames 10      # process frames 0 through 9
                    set frames 1:10    # process frames 1 through 9
                    set frames eof     # process all frames till end-of-file
                    set frames live    # input is live until terminated by user
                    set frames default # process all frames specified on input
            set dump-profile [on|off]
                Turn on/off profiler output.
            set wait [key|<milliseconds>]
                Specify wait time between frame processing to give extra time
                for viewing. Or wait for key press between frames.
            set compare [on|off|discard-errors]
                Turn on/off data compares or just discard data compare errors.
            set use-schedule-graph [on|off]
                Turn on/off use of vxScheduleGraph instead of vxProcessGraph.
            set dump-data-config [<dumpFilePrefix>,<obj-type>[,<obj-type>[...]]]
                Specify dump data config for portion of the graph. To disable
                don't specify any config.

    graph <command> [<arguments> ...]
        Specify below graph specific commands:
            graph auto-age [<delayName> [<delayName> ...]]
                Make the default graph use vxAgeDelay API for the specified
                delay objects after processing each frame.
            graph affinity [CPU|GPU[<device-index>]]
                Specify graph affinity to CPU or GPU.
            graph save-and-reset <graphName>
                Verify the default graph and save it as <graphName>. Then
                create a new graph as the default graph. Note that the earlier
                virtual data object won't be available after graph reset.
            graph reset [<graphName(s)>]
                Reset the default or specified graph(s). Note that the earlier
                virtual data object won't be available after graph reset.
            graph launch [<graphName(s)>]
                Launch the default or specified graph(s).
            graph info [<graphName(s)>]
                Show graph details for debug.

    rename <dataNameOld> <dataNameNew>
        Rename a data object\n

    init <dataName> <initial-value>
        Initialize data object with specified value.
        - convolution object initial values can be:
            {<value1>;<value2>;...<valueN>}
            scale{<scale>}
        - matrix object initial values can be:
            {<value1>;<value2>;...<valueN>}
        - remap object initial values can be:
            dst is same as src: same
            dst is 90 degree rotation of src: rotate-90
            dst is 180 degree rotation of src: rotate-180
            dst is 270 degree rotation of src: rotate-270
            dst is horizontal flip of src: hflip
            dst is vertical flip of src: vflip
        - threshold object initial values can be:
            For VX_THRESHOLD_TYPE_BINARY: <value>
            For VX_THRESHOLD_TYPE_RANGE: {<lower>;<upper>}
        - image object initial values can be:
            Binary file with image data. For images created from handle,
            the vxSwapHandles API will be invoked before executing the graph.
        - tensor object initial values can be:
            Binary file with tensor data.
            To replicate a file multiple times, use @repeat~N~<fileName>.
            To fill the tensor with a value, use @fill~f32~<float-value>,
            @fill~i32~<int-value>, @fill~i16~<int-value>, or @fill~u8~<uint-value>.

    read <dataName> <fileName> [ascii|binary] [<option(s)>]
        Read frame-level data from the specified <fileName>.
        - images can be read from containers (such as, .jpg, .avi, .mp4, etc.)
        as well as raw binary files
        - certain raw data formats support reading data for all frames from a
        single file (such as, video.yuv, video.rgb, video.avi etc.)
        The data objects that support this feature are image, scalar, and
        threshold data objects.
        - certain data formats support printf format-syntax (e.g., joy_%04d.yuv)
        to read individual data from separate files. Note that scalar and
        threshold data objects doesn't support this feature. Also note that
        pyramid objects expect all frames of each level in separate files.
        - convolution objects support the option: scale
        This will read scale value as the first 32-bit integer in file(s).

    write <dataName> <fileName> [ascii|binary] [<option(s)>]
        Write frame-level data to the specified <fileName>.
        - certain raw data formats support writing data for all frames into a
        single file (such as, video.yuv, video.rgb, video.u8, etc.)
        The data objects that support this feature are image, scalar, and
        threshold data objects.
        - certain data formats support printf format-syntax (e.g., joy_%04d.yuv)
        to write individual data from separate files. Note that scalar and
        threshold data objects doesn't support this feature. Also note that
        pyramid objects expect all frames of each level in separate files.
        - convolution objects support the option: scale
        This will write scale value as the first 32-bit integer in file(s).

    compare <dataName> <fileName> [ascii|binary] [<option(s)>]
        Compare frame-level data from the specified <fileName>.
        - certain raw data formats support comparing data for all frames from a
        single file (such as, video.yuv, video.rgb, video.u8, etc.)
        The data objects that support this feature are image, scalar, and
        threshold data objects.
        - certain data formats support printf format-syntax (e.g., joy_%04d.yuv)
        to read individual data from separate files. Note that scalar and
        threshold data objects doesn't support this feature.
        - array objects with VX_TYPE_KEYPOINT data type support the options:
            specify tolerance: err{<x>;<y>;<strength>[;<%mismatch>]}
            specify compare log file: log{<fileName>}
        - array objects with VX_TYPE_COORDINATES2D data type support the options:
            specify tolerance: err{<x>;<y>[;<%mismatch>]}
            specify compare log file: log{<fileName>}
        - convolution objects support the option:
            read scale value as the first 32-bit integer in file(s): scale
        - image and pyramid objects support the options:
            specify compare region: rect{<start-x>;<start-y>;<end-x>;<end-y>}
            specify valid pixel difference: err{<min>;<max>}
            specify pixel checksum to compare: checksum
            specify generate checksum: checksum-save-instead-of-test
        - matrix objects support the options:
            specify tolerance: err{<tolerance>}
        - remap objects support the options:
            specify tolerance: err{<x>;<y>}
        - scalar objects support the option:
            specify that file specifies inclusive range of valid values: range

    view <dataName> <windowName>
        Display frame-level data in a window with title <windowName>. Each window
        can display an image data object and optionally additional other data
        objects overlaid on top of the image.
        - supported data object types are: array, distribution, image, lut,
        scalar, and delay.
        - display of array, distribution, lut, and scalar objects are
        overlaid on top of an image with the same <windowName>.
        - delay object displays reference in the slot#0 of current time.

    directive <dataName> <directive>
        Specify a directive to data object. Only a few directives are supported:
        - Use sync-cl-write directive to issue VX_DIRECTIVE_AMD_COPY_TO_OPENCL
        directive whenever data object is updated using init or read commands.
        Supported for array, image, lut, and remap data objects only.
        - Use readonly directive to issue VX_DIRECTIVE_AMD_READ_ONLY directive
        that informs the OpenVX framework that object won't be updated after
        init command. Supported for convolution and matrix data objects only.

    pause
        Wait until a key is pressed before processing next GDF command.

    help <command>
        Show the GDF command help.

    exit
        Exit from shell or included GDF file.

    quit
        Abort the application.


Examples
========

The following are examples that demonstrate the use of RUNVX prototyping tool.

Canny Edge Detector
-------------------

This example demonstrates building OpenVX graph for Canny edge detector. Use [face1.jpg](https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/face1.jpg) for this example.

.. code-block:: shell

    % runvx[.exe] file canny.gdf

The contents of ``canny.gdf``:

.. code-block:: shell

    # create input and output images
    data input  = image:480,360,RGB2
    data output = image:480,360,U008
    
    # specify input source for input image and request for displaying input and output images
    read input  examples/images/face1.jpg
    view input  inputWindow
    view output edgesWindow
    
    # compute luma image channel from input RGB image
    data yuv  = image-virtual:0,0,IYUV
    data luma = image-virtual:0,0,U008
    node org.khronos.openvx.color_convert input yuv
    node org.khronos.openvx.channel_extract yuv !CHANNEL_Y luma
    
    # compute edges in luma image using Canny edge detector
    data hyst = threshold:RANGE,U008,U008:INIT,80,100
    data gradient_size = scalar:INT32,3
    node org.khronos.openvx.canny_edge_detector luma hyst gradient_size !NORM_L1 output


Skintone Pixel Detector
-----------------------

This example demonstrates building OpenVX graph for pixel-based skin tone detector [Peer et al. 2003]. Use [face1.jpg](https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/face1.jpg) for this example.

.. code-block:: shell

    % runvx[.exe] file skintonedetect.gdf

The contents of ``skintonedetect.gdf``:

.. code-block:: shell

    # create input and output images
    data input  = image:480,360,RGB2
    data output = image:480,360,U008

    # specify input source for input image and request for displaying input and output images
    read input  examples/images/face1.jpg
    view input  inputWindow
    view output skintoneWindow

    # threshold objects
    data thr95  = threshold:BINARY,U008,U008:INIT,95 # threshold for computing R > 95
    data thr40  = threshold:BINARY,U008,U008:INIT,40 # threshold for computing G > 40
    data thr20  = threshold:BINARY,U008,U008:INIT,20 # threshold for computing B > 20
    data thr15  = threshold:BINARY,U008,U008:INIT,15 # threshold for computing R-G > 15
    data thr0   = threshold:BINARY,U008,U008:INIT,0  # threshold for computing R-B > 0

    # virtual image objects for intermediate results
    data R      = image-virtual:0,0,U008
    data G      = image-virtual:0,0,U008
    data B      = image-virtual:0,0,U008
    data RmG    = image-virtual:0,0,U008
    data RmB    = image-virtual:0,0,U008
    data R95    = image-virtual:0,0,U008
    data G40    = image-virtual:0,0,U008
    data B20    = image-virtual:0,0,U008
    data RmG15  = image-virtual:0,0,U008
    data RmB0   = image-virtual:0,0,U008
    data and1   = image-virtual:0,0,U008
    data and2   = image-virtual:0,0,U008
    data and3   = image-virtual:0,0,U008

    # extract R,G,B channels and compute R-G and R-B
    node org.khronos.openvx.channel_extract input !CHANNEL_R R # extract R channel
    node org.khronos.openvx.channel_extract input !CHANNEL_G G # extract G channel
    node org.khronos.openvx.channel_extract input !CHANNEL_B B # extract B channel
    node org.khronos.openvx.subtract R   G   !SATURATE RmG  # compute R-G
    node org.khronos.openvx.subtract R   B   !SATURATE RmB  # compute R-B

    # compute threshold
    node org.khronos.openvx.threshold R   thr95 R95         # compute R > 95
    node org.khronos.openvx.threshold G   thr40 G40         # compute G > 40
    node org.khronos.openvx.threshold B   thr20 B20         # compute B > 20
    node org.khronos.openvx.threshold RmG thr15 RmG15       # compute RmG > 15
    node org.khronos.openvx.threshold RmB thr0  RmB0        # compute RmB > 0

    # aggregate all thresholded values to produce SKIN pixels
    node org.khronos.openvx.and R95   G40   and1            # compute R95 & G40
    node org.khronos.openvx.and and1  B20   and2            # compute B20 & and1
    node org.khronos.openvx.and RmG15 RmB0  and3            # compute RmG15 & RmB0
    node org.khronos.openvx.and and2 and3 output            # compute and2 & and3 as output


Feature Tracker
---------------

The feature tracker example demonstrates building an application with two separate graphs that uses Harris Corners and Optical Flow kernels.
This example requires use of delay data objects that contain multiple pyramid and array objects.
Use `PETS09-S1-L1-View001.avi <http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi>`_ as input video sequence.

.. code-block:: shell

    % runvx[.exe] file feature_tracker.gdf

The contents of ``feature_tracker.gdf``:

.. code-block:: shell

    # create image object for the input video sequence.
    data input = image:768,576,RGB2
    read input PETS09-S1-L1-View001.avi
    
    # create output keypoint array objects inside a delay object with two slots.
    # two slots are needed to keep track current keypoints from previous time.
    data exemplarArr = array:KEYPOINT,10000   # max trackable keypoints are 10,000
    data delayArr = delay:exemplarArr,2       # two slots inside the delay object
    
    # request for displaying input with keypoints from delay slot[0].
    view input    feature_tracker
    view delayArr feature_tracker
    
    # create pyramid objects inside a delay object with two slots.
    # two slots of pyramids are needed for optical flow kernel.
    data exemplarPyr = pyramid:6,half,768,576,U008
    data delayPyr = delay:exemplarPyr,2

    # create first graph to initialize keypoints using Harris Corners and
    # compute pyramid for by Optical Flow later using another graph
    data iyuv = image-virtual:0,0,IYUV
    data luma = image-virtual:0,0,U008
    data strength_thresh = scalar:FLOAT32,0.0005
    data min_distance = scalar:FLOAT32,5.0
    data sensitivity = scalar:FLOAT32,0.04
    data grad_size = scalar:INT32,3
    data block_size = scalar:INT32,3
    node org.khronos.openvx.color_convert    input  iyuv
    node org.khronos.openvx.channel_extract  iyuv !CHANNEL_Y luma
    node org.khronos.openvx.harris_corners   luma strength_thresh min_distance sensitivity \
                                             grad_size block_size delayArr[0] null
    node org.khronos.openvx.gaussian_pyramid luma delayPyr[0]

    # request vxAgeDelay call for delay objects after each frame with
    # current graph and save current graph with the name "harris"
    graph auto-age delayPyr delayArr
    graph save-and-reset harris

    # create second graph to track keypoints using Optical Flow assuming that
    # pyramid/keypoints in delay objects have been initialized with previous frame
    data iyuv = image-virtual:0,0,IYUV
    data luma = image-virtual:0,0,U008
    data termination = scalar:ENUM,CRITERIA_BOTH
    data epsilon = scalar:FLOAT32,0.01
    data num_iterations = scalar:UINT32,5
    data use_initial_estimate = scalar:BOOL,0
    data window_dimension = scalar:SIZE,6
    node org.khronos.openvx.color_convert       input  iyuv
    node org.khronos.openvx.channel_extract     iyuv !CHANNEL_Y luma
    node org.khronos.openvx.gaussian_pyramid    luma delayPyr[0]
    node org.khronos.openvx.optical_flow_pyr_lk delayPyr[-1] delayPyr[0] \
                                                delayArr[-1] delayArr[-1] delayArr[0] \
                                                termination epsilon num_iterations \
                                                use_initial_estimate window_dimension

    # request vxAgeDelay call for delay objects after each frame with
    # current graph and save current graph with the name "opticalflow"
    graph auto-age delayPyr delayArr
    graph save-and-reset opticalflow

    # launch "harris" graph to process first frame in the video sequence
    set frames 1
    graph launch harris

    # launch "opticalflow" graph to process remaining frames in the video sequence
    set frames default
    graph launch opticalflow

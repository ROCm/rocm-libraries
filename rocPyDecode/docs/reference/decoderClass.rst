.. meta::
  :description: rocPyDecode Decoder class documentation
  :keywords: rocPyDecode, rocDecode, ROCm, API, documentation, video, decode, decoding, acceleration

.. _decoder_section:

*************************
rocPyDecode decoder class
*************************

Instantiation
=============

To instantiate a decoder instance, pass the following parameters:  

- **device_id**:	The GPU device ID, default is 0
- **mem_type**:		The mem type of output surface - 0: Internal 1: dev_copied 2: host_copied, default is 1
- **codec**:        The codec ID obtained by *GetRocDecCodecID* API
- **b_force_zero_latency**: 	Force zero latency flag, default is 'False'
- **crop_rect**:    See :doc:`Structures Classess <structures>`, optional, default: 'None', no cropping
- **max_width**:    Max width, default is 0
- **max_height**:   Max height, default is 0
- **clk_rate**:     Clock rate, default is 1000  

Example:
--------

.. code-block:: shell

	# create decoder instance
	viddec = dec.decoder(device_id, mem_type, codec_id, b_force_zero_latency, crop_rect, 0, 0, 1000)  

Member functions
================

The following are the member functions of the Decode class.

DecodeFrame(packet)
-------------------

Decodes video frames described by the input :ref:`packet` obtained from demux functions. Returns the count of decoded frames.


GetFrame(packet)
----------------

Obtains a pointer to the current decoded frame in the input packet. Returns the current frame time stamp, the frame pointer, and the time stamp saved in the packet.

Example:
^^^^^^^^

.. code-block:: python
	
		# decoding loop
		while True:
			packet = demuxer.DemuxFrame()

			n_frame_returned = viddec.DecodeFrame(packet)

			for i in range(n_frame_returned):
				viddec.GetFrame(packet)

			# TO DO: process the frame or save it, etc.
			# ...

			# release frame
			viddec.ReleaseFrame(packet)

			if (packet.frame_size <= 0):  # EOF: no more to decode
				break

				
GetFrameRgb(packet, rgb_format)
-------------------------------

Obtains a pointer to the current decoded frame in the input packet, and converts that YUV frame to 'Tensor' in RGB format. Returns the current frame time stamp, the frame pointer, and the time stamp saved in the packet.

- **packet**: The demuxed packet contains the demuxed frames information, and the desired rgb format
- **rgb_format**: 1 for bgr, 3 for rgb

GetFrameSize()
--------------

Returns the size of the current decoded frame in bytes.

GetGpuInfo()
------------

Returns the :ref:`configinfo` for the current GPU device

Example:
^^^^^^^^

.. code-block:: python
	
		# Get GPU device information
		cfg = viddec.GetGpuInfo()
		
		# print GPU info out
		print("GPU device " + str(device_id) + " \- " + cfg.device_name + "[" + cfg.gcn_arch_name + "] on PCI bus " + str(cfg.pci_bus_id) + ":" + str(cfg.pci_domain_id) + "." + str(cfg.pci_device_id))


GetHeight()
-----------

Returns the height of the current decoded frame.

GetOutputSurfaceInfo()
----------------------

Returns the current decoded frame's surface information structure pointer.

GetResizedOutputSurfaceInfo()
-----------------------------

Returns the current decoded and resized frame's surface information structure pointer.

GetStride()
-----------

Returns the stride of the current decoded frame.

GetWidth()
----------

Returns the width of the current decoded frame.

ResizeFrame(packet, resize_dim, surface_info)
---------------------------------------------

Resizes the decode frame pointed to by the passed packet, see: :doc:`Structures Classess <structures>`, to the new dimension in resize_dim, to the new dimension using the passed surface info.

- **packet**: The demuxed packet contains the demuxed frames information, and the desired rgb format  
- **resize_dim**:  The new dimension, width and height 
- **surface_info**: The current surface info obtained by GetOutputSurfaceInfo API

Example:
^^^^^^^^

.. code-block:: python
	
		# resize frame to new dimension
		resize__dim = [1024, 720]

		surface__info = viddec.GetOutputSurfaceInfo()

		frame_is_resized = False

		if(viddec.ResizeFrame(packet, resize_dim, surface_info) != 0):
			frame_is_resized = True

SaveFrameToFile(output_file_path, frame_adrs, surface_info)
-----------------------------------------------------------

Saves all the decoded frames to a disk file in YUV format.

- **output_file_path**: The full path disk file name to save the YUV frames
- **frame_adrs**: The current frame pointer, obtained from the used packet
- **surface_info**: The current decode frame surface information structure pointer

SaveTensorToFile(output_file_path, frame_adrs, width, height, rgb_format, surface_info)
---------------------------------------------------------------------------------------

Saves all the decoded frames after being converted to a Tensor to a disk file in RGB format.

- **output_file_path**: The full path disk file name to save the YUV frames
- **frame_adrs**: The current frame/tensor pointer, obtained from the used packet
- **width**: The width of the current Tensor
- **height**: The height of the current Tensor
- **rgb_format**: 1 for bgr, 3 for rgb 
- **surface_info**: The current decode frame surface information structure pointer

ReleaseFrame(packet)
--------------------

Release the GPU memory of the current decoded frame.

GetNumOfFlushedFrames()
-----------------------

Returns the count of the flushed frames.  

Example:
^^^^^^^^

.. code-block:: python
	
		# beyond the decoding loop
		n_frame += viddec.GetNumOfFlushedFrames()  
	
SetReconfigParams(flush_mode, out_file_name)
--------------------------------------------

Specify the flush mode and the output file name to use in multi resolution video support.  

- **flush_mode**: 

	- 0: Just flush to get the frame count
	- 1: The remaining frames will be dumped to file in this mode

- **out_file_name**: The full path disk file name to save the YUV frames

Example:
^^^^^^^^

.. code-block:: python
	
		# set reconfiguration params based on user arguments
		flush_mode = 0

		if (output_file_path is not None):
			flush_mode = 1

		viddec.SetReconfigParams(flush_mode, output_file_path if (output_file_path is not None) else str(""))

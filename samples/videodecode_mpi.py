# Copyright © Advanced Micro Devices, Inc., or its affiliates.
 
# SPDX-License-Identifier:  MIT License

import torch
import pyRocVideoDecode.decoder as dec
import pyRocVideoDecode.demuxer as dmx
import datetime
import sys
import argparse
import os.path
from mpi4py import MPI	
import subprocess
import av
import numpy as np

# This example uses MPI to distribute rocPyDecode decoding task, each node to decode
# and save an equal section of the total count of frames from the input video file.
# The example also uses MPI-IO to allow each node to save its portion of the
# task in the output file utilizing the built it IO mechanism MPI offers.

def Decoder(
        input_file_path,
        output_file_path
        ):

    # mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # TODO (@rocDecode): -> Get count of frames:
    # The rocDecode demuxer/parser needs to go through the  
    # entire stream without decoding, to get the count of
    # frames. Until such an API being implemented, use av for now

    # Get count of frames
    if rank == 0:        
        container = av.open(input_file_path)               # Open the video file
        frames_total = container.streams.video[0].frames   # Get total number of frames from first video stream
    else:
        frames_total = None

    comm.barrier() # synchronize all processes

    # Create demuxer instance
    demuxer = dmx.demuxer(input_file_path)
    # Get the codec id
    codec_id = dec.GetRocDecCodecID(demuxer.GetCodecId())
    # Create decoder instance
    viddec = dec.decoder(codec_id)
    # Check if codec is supported, exit otherwise
    if (viddec.IsCodecSupported(0, codec_id, demuxer.GetBitDepth()) == False):
        print("ERROR: Codec is not supported on this GPU " + viddec.GetGpuInfo().device_name)
        exit()

    # ---------
    # MPI Setup
    # ---------
 
    # TO: all ranks to know what is 'frames_total' FROM: node 0
    frames_total = comm.bcast(frames_total, root=0)
 
     # THIS rank portion of the task, calc that
    frames_remain = frames_total % size
    each_proc_frame_count = int((frames_total-frames_remain)/size)
    this_rank_start = rank * each_proc_frame_count
    this_rank_end  = this_rank_start + each_proc_frame_count - 1 

    # Add 'frames_remain' to last rank
    if (rank == (size-1)):
        this_rank_end += frames_remain # last rank takes care of the residue

    # Set reconfiguration params if using -o
    save_frames = False
    if output_file_path is not None:
        save_frames = True
        flush_mode = 1
        viddec.SetReconfigParams(flush_mode, output_file_path)

    # -----------------
    # The decoding loop
    # -----------------

    frame_index = 0
    total_dec_time = 0.0
    save_file_prerequisite = False
    saved_frames_index = this_rank_start

    while True:

        start_time = datetime.datetime.now()
        packet = demuxer.DemuxFrame()
        n_of_frames_returned = viddec.DecodeFrame(packet)

        for i in range(n_of_frames_returned):

            viddec.GetFrameYuv(packet, False)
            frame_data = torch.from_dlpack(packet.ext_buf[0].__dlpack__(packet)).cpu().numpy()

            # only save when requested, and only save the assigned frames to this rank
            if (save_frames and (frame_index>=this_rank_start) and (saved_frames_index<=this_rank_end)):
                # Setup 'Save Frame to File' Configuration One Time
                if save_file_prerequisite is not True:
                    width = float(viddec.GetStride())          # Width of each frame in pixels
                    height = float(viddec.GetHeight())         # Height of each frame in pixels
                    bit_depth = float(viddec.GetBitDepth())    # Bit depth of each pixel (e.g., 8, 16, 32)
                    frame_size_bytes = int((width * height * 1.5 * bit_depth) // 8.0)  # Calculate frame size in bytes, 1.5 for the UV to be included in NV12 format
                    save_file_prerequisite = True
                    # Open the file for writing in parallel
                    try:
                        f = MPI.File.Open(comm, output_file_path, MPI.MODE_WRONLY | MPI.MODE_CREATE)
                    except Exception as e:
                        print(f"Process {rank} encountered an error: {e}")

                # Calculate offset based on frame number
                offset = saved_frames_index * frame_size_bytes
                f.Write_at(offset, frame_data.ravel()) # Independent File Pointer
                saved_frames_index += 1

            # release frame
            viddec.ReleaseFrame(packet)

        # Measure time after completing a whole frame
        end_time = datetime.datetime.now()
        time_per_frame = end_time - start_time
        total_dec_time = total_dec_time + time_per_frame.total_seconds()

        # Increment frames counter
        frame_index += n_of_frames_returned

        # Stop if no more to demux/decode
        if (packet.bitstream_size <= 0):  # EOF: no more to decode
            break

    # close file if we have opened it
    if save_frames:
        f.Close()

    # beyond the decoding loop
    frame_index += viddec.GetNumOfFlushedFrames()        

    # collect total elapsed time in all processes
    total_time_elapsed = comm.gather(total_dec_time, root=0)

    # done using MPI
    MPI.Finalize

    # report time
    if (rank == 0):
        frame_index = frame_index * size # we decoded 'size' times, depends on the -n mpirun/mpiexec param
        if (total_time_elapsed is not None):
            total_dec_time = sum(total_time_elapsed)/size # average of all processes time
        time_per_frame = (total_dec_time / frames_total) * 1000 # we decoded ALL anyway
        frame_per_second = frame_index / total_dec_time

        print("\ninfo: avg decoding time per frame: " +"{0:0.2f}".format(round(time_per_frame, 2)) + " ms")
        print("info: avg frame per second: " +"{0:0.2f}".format(round(frame_per_second,2)) +"\n")

        print(f"Total nodes count executed: {size}\n")

        # for user dbg
        # if (total_time_elapsed is not None):
        #     print(f"\nElapsed time reported by all processes: {total_time_elapsed}")
        #     print(f"Highest elapsed time (actual decoding time): {max(total_time_elapsed)}")
        # print("Total processors: ", size)
        # print("Frames per proc:  ", each_proc_frame_count)
        # print("Frames Remaining: ", frames_remain)

if __name__ == "__main__":

    # get passed arguments
    parser = argparse.ArgumentParser(
        description='PyRocDecode Video Decode Arguments')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input File Path - required',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output File Path - optional',
        required=False)

    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    # get params
    input_file_path = args.input
    output_file_path = args.output

    # handle params
    if not os.path.exists(input_file_path):  # Input file (must exist)
        print("ERROR: input file doesn't exist.")
        exit()

    # call main routine
    Decoder(
        input_file_path,
        output_file_path
    )

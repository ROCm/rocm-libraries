# Copyright © Advanced Micro Devices, Inc., or its affiliates.
 
# SPDX-License-Identifier:  MIT License

import pyRocVideoDecode.decoder as dec
import pyRocVideoDecode.demuxer as dmx
import datetime
import sys
import argparse
import os.path


def Decoder(
        input_file_path,
        output_file_path,
        device_id,
        mem_type,
        b_force_zero_latency,
        crop_rect,
        seek_frame,
        seek_mode,
        seek_criteria,
        resize_dim):

    # create file stream provider object
    stream_provider = dmx.stream_provider(input_file_path)
    # demuxer instance
    demuxer = dmx.demuxer(stream_provider)

    # get the used coded id
    codec_id = dec.GetRocDecCodecID(demuxer.GetCodecId())

    # decoder instance
    viddec = dec.decoder(codec_id, device_id, mem_type)

    # Get GPU device information
    cfg = viddec.GetGpuInfo()

    # check if codec is supported
    if (viddec.IsCodecSupported(device_id, codec_id, demuxer.GetBitDepth()) == False):
        print("ERROR: Codec is not supported on this GPU " + cfg.device_name)
        exit()

    #  print some GPU info out
    print("\ninfo: Input file: " +
          input_file_path +
          '\n' +
          "info: Using GPU device " +
          str(device_id) +
          " - " +
          cfg.device_name +
          "[" +
          cfg.gcn_arch_name +
          "] on PCI bus " +
          str(cfg.pci_bus_id) +
          ":" +
          str(cfg.pci_domain_id) +
          "." +
          str(cfg.pci_device_id))
    print("info: decoding started, please wait! \n")

    # set reconfiguration params based on user arguments
    flush_mode = 0
    if (output_file_path is not None):
        flush_mode = 1

    viddec.SetReconfigParams(flush_mode, output_file_path if (output_file_path is not None) else str(""))

    # -----------------
    # The decoding loop
    # -----------------
    n_frame = 0
    total_dec_time = 0.0
    frame_is_resized = False
    not_seeking = True if (seek_frame == -1) else False
    session_id = 0

    if (resize_dim is not None):
        resize_dim = None if(resize_dim[0] == 0 or resize_dim[1] == 0) else resize_dim

    while True:
        start_time = datetime.datetime.now()

        if(not_seeking):
            packet = demuxer.DemuxFrame()
        else:
            packet = demuxer.SeekFrame(seek_frame, seek_mode, seek_criteria)
            not_seeking = True

        n_frame_returned = viddec.DecodeFrame(packet)

        for i in range(n_frame_returned):
            viddec.GetFrameYuv(packet)

            if (resize_dim is not None):
                surface_info = viddec.GetOutputSurfaceInfo()
                if(viddec.ResizeFrame(packet, resize_dim, surface_info) != 0):
                    frame_is_resized = True
                else:
                    frame_is_resized = False

            if (output_file_path is not None):
                if (frame_is_resized):
                    resized_surface_info = viddec.GetResizedOutputSurfaceInfo()
                    viddec.SaveFrameToFile(output_file_path, packet.frame_adrs_resized, resized_surface_info)
                else:
                    viddec.SaveFrameToFile(output_file_path, packet.frame_adrs)

            # release frame
            viddec.ReleaseFrame(packet)

        # measure after completing a whole frame
        end_time = datetime.datetime.now()
        time_per_frame = end_time - start_time
        total_dec_time = total_dec_time + time_per_frame.total_seconds()

        # increament frames counter
        n_frame += n_frame_returned

        if (packet.bitstream_size <= 0):  # EOF: no more to decode
            break

    # beyond the decoding loop
    n_frame += viddec.GetNumOfFlushedFrames()

    print("info: Total frame decoded: " + str(n_frame))

    if (output_file_path is None):
        if (n_frame > 0 and total_dec_time > 0):
            time_per_frame = (total_dec_time / n_frame) * 1000
            session_overhead = viddec.GetDecoderSessionOverHead(session_id)
            if (session_overhead == None):
                session_overhead = 0
            time_per_frame -= (session_overhead / n_frame) # remove the overhead
            frame_per_second = n_frame / total_dec_time
            print("info: avg decoding time per frame: " +"{0:0.2f}".format(round(time_per_frame, 2)) + " ms")
            print("info: avg frame per second: " +"{0:0.2f}".format(round(frame_per_second,2)) +"\n")
        else:
            print("info: frame count= ", n_frame)

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
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
        help='GPU device ID - optional, default 0',
        required=False)
    parser.add_argument(
        '-m',
        '--mem_type',
        type=int,
        default=1,
        help='mem_type of output surfce - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped, optional, default 0',
        required=False)
    parser.add_argument(
        '-z',
        '--zero_latency',
        type=str,
        default='no',
        choices=['yes', 'no'],
        help='Force zero latency',
        required=False)
    parser.add_argument(
        '-crop',
        '--crop_rect',
        nargs=4,
        type=int,
        help='Crop rectangle (left, top, right, bottom), optional, default: no cropping',
        required=False)
    parser.add_argument(
        '-s',
        '--seek',
        type=int,
        default=-1,
        help='seek this number of frames, optional, default: no seek',
        required=False)
    parser.add_argument(
        '-sm',
        '--seek_mode',
        type=int,
        default=1,
        help='seek mode, 0 - by exact frame number, 1 - by previous key frame, optional, default: 1 - by previous key frame',
        required=False)
    parser.add_argument(
        '-sc',
        '--seek_criteria',
        type=int,
        default=0,
        help='seek criteria, 0 - by frame number, 1 - by time stamp, optional, default: 0 - by frame number',
        required=False)
    parser.add_argument(    
        '-resize',
        '--resize_dim',
        nargs=2,
        type=int,
        help='Width & Height of new frame, optional, default: no resizing',
        required=False)
    
    try:
        args = parser.parse_args()
    except BaseException:
        sys.exit()

    # get params
    input_file_path = args.input
    output_file_path = args.output
    device_id = args.device
    mem_type = args.mem_type
    b_force_zero_latency = args.zero_latency.upper()
    crop_rect = args.crop_rect
    seek_frame = args.seek
    seek_mode = args.seek_mode
    seek_criteria = args.seek_criteria
    resize_dim = args.resize_dim
    
    # validate the seek: mode/criteria
    if(seek_frame > 0):
        if(seek_mode != 0 and seek_mode != 1):
            print("Error: Invalid seek mode value.")
            exit()
        if(seek_criteria != 0 and seek_criteria != 1):
            print("Error: Invalid seek criteria value.")
            exit()

    # handle params
    mem_type = 0 if (mem_type < 0 or mem_type > 3) else mem_type
    b_force_zero_latency = True if b_force_zero_latency == 'YES' else False
    if not os.path.exists(input_file_path):  # Input file (must exist)
        print("ERROR: input file doesn't exist.")
        exit()

    Decoder(
        input_file_path,
        output_file_path,
        device_id,
        mem_type,
        b_force_zero_latency,
        crop_rect,
        seek_frame,
        seek_mode,
        seek_criteria,
        resize_dim)
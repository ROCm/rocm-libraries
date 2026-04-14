/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/mlo_internal.hpp>
#include <miopen/logger.hpp>
#include <numbers>

// KNOWN ISSUES:
// backward propogagation has a bug in cross map normalization when numper of maps less than
// normalization region

void mlo_construct_norm::mloConstruct()
{
    if(_problem.IsDirectionForward())
    {
        mloConstructFwd();
    }
    else
    {
        mloConstructBwd();
    }
}

inline bool is_tensor_packed(int c, int h, int w, int b_str, int c_str, int h_str)
{
    return h_str == w && c_str == h * h_str && b_str == c * c_str;
}

void mlo_construct_norm::mloConstructFwd()
{
    size_t maxComputeUnits = _ctx.GetStream().GetMaxComputeUnits();

    _hw_wave_sz = 64;

    int pre_pad = (_norm_area - 1) / 2;
    int pad     = _norm_area - pre_pad - 1;

    if(pre_pad < 0 || pad < 0)
        MIOPEN_THROW("Wrong LRN kernel size");

    _grp_tile0     = (_problem.GetOutWidth() <= 16) ? 8 : 16;
    _grp_tile1     = 8;
    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;

    auto is_in_packed = is_tensor_packed(_problem.GetInChannels(),
                                         _problem.GetInHeight(),
                                         _problem.GetInWidth(),
                                         _problem.GetInBatchStride(),
                                         _problem.GetInChannelStride(),
                                         _problem.GetInStride());

    int map_size_4 = _problem.GetInWidth() * (is_in_packed ? _problem.GetInHeight() : 1);
    int read_unit;
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0 = (_problem.GetOutWidth() <= 8) ? 8 : 16;
        _grp_tile1 = (_problem.GetOutHeight() <= 8) ? 8 : 16;
        read_unit  = (map_size_4 % 4 == 0) ? 4 : (map_size_4 % 2 == 0) ? 2 : 1;
        map_size_4 /= read_unit;
    }
    else
    {

        _out_pix_tile0 = (_problem.GetOutWidth() <= 8) ? 1 : 2;
        _out_pix_tile1 = (_problem.GetOutHeight() <= 8) ? 1 : 2;
        read_unit      = 4;
        map_size_4     = (map_size_4 + 3) / 4;
    }
    map_size_4 *= (is_in_packed ? 1 : _problem.GetInHeight());

    assert(_out_pix_tile0 - 1 <= _norm_area && _out_pix_tile1 - 1 <= _norm_area);

    _kernel_file = "MIOpenLRNFwd.cpp";
    _kernel_name = (_norm_region == MLO_LRN_ACROSS_CHANNELS) ? "MIOpenLRNAcrossChannels4"
                                                             : "MIOpenLRNWithinChannel_PS";
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0  = 8 * 8;
        _grp_tile1  = 1;
        int n_waves = (_problem.GetBatchSize() * map_size_4 + _hw_wave_sz - 1) / _hw_wave_sz;
        if(n_waves <= maxComputeUnits * 8)
        {
            map_size_4 = _problem.GetInWidth() * (is_in_packed ? _problem.GetInHeight() : 1);
            read_unit  = (map_size_4 % 2 == 0) ? 2 : 1;
            map_size_4 /= read_unit;
            map_size_4 *= (is_in_packed ? 1 : _problem.GetInHeight());
        }
    }

    // Workaround for ROCm 1.8.2 compiler issue (#1057).
    if(_problem.GetInDataType() == miopenHalf && read_unit > 1 &&
       _kernel_name == "MIOpenLRNAcrossChannels4")
    {
        const std::string name = _ctx.GetStream().GetDeviceName();
        if(name.find("gfx9") != std::string::npos) // Any gfx9 device.
        {
            MIOPEN_LOG_I("Workaround for #1057: "
                         << name << ',' << miopen::GetDataTypeName(_problem.GetInDataType()) << ','
                         << map_size_4 << ',' << read_unit);
            map_size_4 *= read_unit;
            read_unit = 1;
        }
    }

    int scale_stride         = _problem.GetOutStride();
    int scale_channel_stride = _problem.GetOutChannelStride();
    int scale_batch_stride   = _problem.GetOutBatchStride();
    int scale                = (doBackward()) ? 1 : 0;

    auto g_wk_width = static_cast<int>((_problem.GetOutWidth() + _grp_tile0 * _out_pix_tile0 - 1) /
                                       (_grp_tile0 * _out_pix_tile0));
    auto g_wk_height =
        static_cast<int>((_problem.GetOutHeight() + _grp_tile1 * _out_pix_tile1 - 1) /
                         (_grp_tile1 * _out_pix_tile1));
    int out_vert_aligned =
        (g_wk_height * (_grp_tile1 * _out_pix_tile1) == _problem.GetOutHeight()) ? 1 : 0;
    int out_horiz_aligned =
        (g_wk_width * (_grp_tile0 * _out_pix_tile0) == _problem.GetOutWidth()) ? 1 : 0;
    // currently always 1
    bool div_by_4 = (map_size_4 * read_unit == _problem.GetInWidth() * _problem.GetInHeight());
    int c1x1_pixleft =
        div_by_4 ? 0
                 : _problem.GetInWidth() * _problem.GetInHeight() - (map_size_4 - 1) * read_unit;
    _comp_options =
        std::string(" -DKERNEL_SIZE=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DPAD=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DKERNEL_SIZE1=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DPAD0=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DPRE_PAD=") + std::to_string(static_cast<long long>(pre_pad)) +
        std::string(" -DPRE_PAD1=") + std::to_string(static_cast<long long>(pre_pad)) +
        std::string(" -DKERNEL_SIZE0=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DPRE_PAD0=") + std::to_string(static_cast<long long>(pre_pad)) +
        std::string(" -DN_OUTPUTS=") +
        std::to_string(static_cast<long long>(_problem.GetOutChannels())) +
        std::string(" -DN_INPUTS=") +
        std::to_string(static_cast<long long>(_problem.GetInChannels())) +
        std::string(" -DHORIZ_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile0)) +
        std::string(" -DVERT_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile1)) +
        std::string(" -DGROUP_SIZE_X=") + std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DGROUP_SIZE_Y=") + std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(" -DBOT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInBatchStride())) +
        std::string(" -DBOT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInChannelStride())) +
        std::string(" -DBOT_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInStride())) +
        std::string(" -DTOP_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutBatchStride())) +
        std::string(" -DTOP_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutChannelStride())) +
        std::string(" -DTOP_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutStride())) +
        std::string(" -DBOT_WIDTH=") +
        std::to_string(static_cast<long long>(_problem.GetOutWidth())) +
        std::string(" -DBOT_HEIGHT=") +
        std::to_string(static_cast<long long>(_problem.GetOutHeight())) +
        std::string(" -DTOP_WIDTH=") +
        std::to_string(static_cast<long long>(_problem.GetOutWidth())) +
        std::string(" -DTOP_HEIGHT=") +
        std::to_string(static_cast<long long>(_problem.GetOutHeight())) +
        std::string(" -DSCALE_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(scale_batch_stride)) +
        std::string(" -DSCALE_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(scale_channel_stride)) +
        std::string(" -DSCALE_STRIDE=") + std::to_string(static_cast<long long>(scale_stride)) +
        std::string(" -DBATCH_SIZE=") +
        std::to_string(static_cast<long long>(_problem.GetBatchSize())) +
        std::string(" -DDO_SCALE=") + std::to_string(static_cast<long long>(scale)) +
        std::string(" -DVERT_ALIGNED=") + std::to_string(static_cast<long long>(out_vert_aligned)) +
        std::string(" -DHORIZ_ALIGNED=") +
        std::to_string(static_cast<long long>(out_horiz_aligned)) + std::string(" -DMAP_SZ_4=") +
        std::to_string(static_cast<long long>(map_size_4)) + std::string(" -DC1x1_PIXLEFT=") +
        std::to_string(static_cast<long long>(c1x1_pixleft)) + std::string(" -DREAD_UNIT=") +
        std::to_string(static_cast<long long>(read_unit)) + getGeneralCompOptions();

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {

        _g_wk.push_back(map_size_4);
        _g_wk.push_back(1);
        _g_wk.push_back(_problem.GetBatchSize());
    }
    else
    {

        _g_wk.push_back(static_cast<size_t>(g_wk_width) * _grp_tile0);
        _g_wk.push_back(static_cast<size_t>(g_wk_height) * _grp_tile1);
        _g_wk.push_back(static_cast<size_t>(_problem.GetOutChannels()) * _problem.GetBatchSize());
    }
    int data_len = miopen::GetTypeSize(_problem.GetOutDataType());

    // calculate workspace
    size_t scale_sz = static_cast<size_t>(_problem.GetBatchSize()) * scale_batch_stride * data_len;
    _workspace_sz   = (doBackward()) ? scale_sz : 0;
}

void mlo_construct_norm::mloConstructBwd()
{
    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;
    _grp_tile0     = 8;
    _grp_tile1     = 8;
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0 = (_in_df_width <= 8) ? 8 : 16;
        _grp_tile1 = (_in_df_height <= 8) ? 8 : 16;
    }
    else
    {
        _out_pix_tile0 = (_in_df_width <= 8) ? 1 : (_in_df_width <= 16) ? 2 : 4;
        _out_pix_tile1 = (_in_df_height <= 8) ? 1 : (_in_df_height <= 16) ? 2 : 4;
    }

    int pre_pad              = (_norm_area - 1) / 2;
    int pad                  = _norm_area - pre_pad - 1;
    int scale_stride         = _problem.GetOutStride();
    int scale_channel_stride = _problem.GetOutChannelStride();
    int scale_batch_stride   = _problem.GetOutBatchStride();

    if(pre_pad < 0 || pad < 0)
        MIOPEN_THROW("Wrong LRN kernel size");

    _comp_options =
        std::string(" -DKERNEL_SIZE=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DOUT_CHANNELS=") +
        std::to_string(static_cast<long long>(_problem.GetOutChannels())) + std::string(" -DPAD=") +
        std::to_string(static_cast<long long>(pad)) + std::string(" -DPRE_PAD=") +
        std::to_string(static_cast<long long>(pre_pad)) + std::string(" -DHORIZ_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile0)) + std::string(" -DVERT_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile1)) + std::string(" -DGROUP_SIZE_X=") +
        std::to_string(static_cast<long long>(_grp_tile0)) + std::string(" -DGROUP_SIZE_Y=") +
        std::to_string(static_cast<long long>(_grp_tile1)) + std::string(" -DBOT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInBatchStride())) +
        std::string(" -DBOT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInChannelStride())) +
        std::string(" -DBOT_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetInStride())) +
        std::string(" -DTOP_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutBatchStride())) +
        std::string(" -DTOP_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutChannelStride())) +
        std::string(" -DTOP_STRIDE=") +
        std::to_string(static_cast<long long>(_problem.GetOutStride())) +
        std::string(" -DBOT_WIDTH=") +
        std::to_string(static_cast<long long>(_problem.GetInWidth())) +
        std::string(" -DBOT_HEIGHT=") +
        std::to_string(static_cast<long long>(_problem.GetInHeight())) +
        std::string(" -DTOP_WIDTH=") +
        std::to_string(static_cast<long long>(_problem.GetOutWidth())) +
        std::string(" -DTOP_HEIGHT=") +
        std::to_string(static_cast<long long>(_problem.GetOutHeight())) +
        std::string(" -DSCALE_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(scale_batch_stride)) +
        std::string(" -DSCALE_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(scale_channel_stride)) +
        std::string(" -DSCALE_STRIDE=") + std::to_string(static_cast<long long>(scale_stride)) +
        std::string(" -DTOPDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_out_df_batch_stride)) +
        std::string(" -DTOPDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_out_df_channel_stride)) +
        std::string(" -DTOPDF_STRIDE=") + std::to_string(static_cast<long long>(_out_df_stride)) +
        std::string(" -DBOTDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_in_df_batch_stride)) +
        std::string(" -DBOTDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_in_df_channel_stride)) +
        std::string(" -DBOTDF_STRIDE=") + std::to_string(static_cast<long long>(_in_df_stride)) +
        std::string(" -DBATCH_SIZE=") +
        std::to_string(static_cast<long long>(_problem.GetBatchSize())) +
        std::string(" -DN_INPUTS=") +
        std::to_string(static_cast<long long>(_problem.GetInChannels())) + getGeneralCompOptions();

    _kernel_file = "MIOpenLRNBwd.cpp";

    _l_wk.clear();
    _g_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _g_wk.push_back(_in_df_width);
        _g_wk.push_back(_in_df_height);
        _g_wk.push_back(_problem.GetBatchSize());
        _kernel_name = "MIOpenLRNAcrossChannelsBwd1";
    }
    else
    {
        int g_wk_width =
            ((_in_df_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
        int g_wk_height =
            ((_in_df_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

        _g_wk.push_back(static_cast<size_t>(g_wk_width) * _grp_tile0);
        _g_wk.push_back(static_cast<size_t>(g_wk_height) * _grp_tile1);
        _g_wk.push_back(static_cast<size_t>(_problem.GetInChannels()) * _problem.GetBatchSize());
        _kernel_name = "MIOpenLRNWithinChannelBwd";
    }
}

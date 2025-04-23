/*
 * Copyright (c) 2012-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OPENVX_VENDORS_H_
#define _OPENVX_VENDORS_H_

/*!
 * \file
 * \brief The Vendor ID list for OpenVX.
 */

/*! \brief The Vendor ID of the Implementation. As new vendors submit their
 * implementations, this enumeration will grow.
 * \ingroup group_basic_features
 */
enum vx_vendor_id_e {
    VX_ID_KHRONOS   = 0x000, /*!< \brief The Khronos Group */
    VX_ID_TI        = 0x001, /*!< \brief Texas Instruments, Inc. */
    VX_ID_QUALCOMM  = 0x002, /*!< \brief Qualcomm, Inc. */
    VX_ID_NVIDIA    = 0x003, /*!< \brief NVIDIA Corporation */
    VX_ID_ARM       = 0x004, /*!< \brief ARM Ltd. */
    VX_ID_BDTI      = 0x005, /*!< \brief Berkley Design Technology, Inc. */
    VX_ID_RENESAS   = 0x006, /*!< \brief Renasas Electronics */
    VX_ID_VIVANTE   = 0x007, /*!< \brief Vivante Corporation */
    VX_ID_XILINX    = 0x008, /*!< \brief Xilinx Inc. */
    VX_ID_AXIS      = 0x009, /*!< \brief Axis Communications */
    VX_ID_MOVIDIUS  = 0x00A, /*!< \brief Movidius Ltd. */
    VX_ID_SAMSUNG   = 0x00B, /*!< \brief Samsung Electronics */
    VX_ID_FREESCALE = 0x00C, /*!< \brief Freescale Semiconductor */
    VX_ID_AMD       = 0x00D, /*!< \brief Advanced Micro Devices */
    VX_ID_BROADCOM  = 0x00E, /*!< \brief Broadcom Corporation */
    VX_ID_INTEL     = 0x00F, /*!< \brief Intel Corporation */
    VX_ID_MARVELL   = 0x010, /*!< \brief Marvell Technology Group Ltd. */
    VX_ID_MEDIATEK  = 0x011, /*!< \brief MediaTek, Inc. */
    VX_ID_ST        = 0x012, /*!< \brief STMicroelectronics */
    VX_ID_CEVA      = 0x013, /*!< \brief CEVA DSP */
    VX_ID_ITSEEZ    = 0x014, /*!< \brief Itseez, Inc. */
    VX_ID_IMAGINATION=0x015, /*!< \brief Imagination Technologies */
    VX_ID_NXP       = 0x016, /*!< \brief NXP Semiconductors */
    VX_ID_VIDEANTIS = 0x017, /*!< \brief Videantis */
    VX_ID_SYNOPSYS  = 0x018, /*!< \brief Synopsys */
    VX_ID_CADENCE   = 0x019, /*!< \brief Cadence */
    VX_ID_HUAWEI    = 0x01A, /*!< \brief Huawei */
    VX_ID_SOCIONEXT = 0x01B, /*!< \brief Socionext */
    /* Add new vendor code above this line */
    VX_ID_USER      = 0xFFE, /*!< \brief For use by vxAllocateUserKernelId and vxAllocateUserKernelLibraryId */
    VX_ID_MAX       = 0xFFF,
    /*! \brief For use by all Kernel authors until they can obtain an assigned ID. */
    VX_ID_DEFAULT = VX_ID_MAX,
};

#endif


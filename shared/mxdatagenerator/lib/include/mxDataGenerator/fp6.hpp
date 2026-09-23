// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

inline uint8_t getDataFromPackedF6(uint8_t const* dataBytes, index_t index)
{
    index_t cellIndex = (index / 4) * 3;

    index_t rem = index % 4;

    uint8_t out = 0b0;

    uint8_t l = 0b0;
    uint8_t r = 0b0;
    switch(rem)
    {
    case 0:
        out = (*(dataBytes + cellIndex) & 0b00111111);
        break;

    case 1:
        l = (*(dataBytes + cellIndex) & 0b11000000) >> 6;
        r = (*(dataBytes + cellIndex + 1) & 0b00001111) << 2;

        out = l | r;

        break;

    case 2:
        l = (*(dataBytes + cellIndex + 1) & 0b11110000) >> 4;
        r = (*(dataBytes + cellIndex + 2) & 0b00000011) << 4;

        out = l | r;

        break;

    case 3:
        out = (*(dataBytes + cellIndex + 2) & 0b11111100) >> 2;
        break;
    }

    return out;
}

inline void setDataPackedF6(uint8_t* dataBytes, size_t index, uint8_t mask)
{
    size_t cellIndex = (index / 4) * 3;

    size_t rem = index % 4;

    uint8_t l = 0b0;
    uint8_t r = 0b0;

    switch(rem)
    {
    case 0:
        *(dataBytes + cellIndex) &= 0b11000000; // blank the last 6 bits
        *(dataBytes + cellIndex) |= mask; // set the mask to the buffer
        break;

    case 1:
        *(dataBytes + cellIndex) &= 0b00111111; // blank the first two bits for first cell
        *(dataBytes + cellIndex + 1) &= 0b11110000; // blank the last 4 bits for the second cell

        l = (mask & 0b00000011) << 6; // get the last two bit, shift it to the left
        r = (mask & 0b00111100) >> 2; // get the first four bit, shift it to the right

        *(dataBytes + cellIndex) |= l; //set the buffers
        *(dataBytes + cellIndex + 1) |= r;

        break;
    case 2:
        *(dataBytes + cellIndex + 1) &= 0b00001111; //blank first 4 bit from first cell
        *(dataBytes + cellIndex + 2) &= 0b11111100; // blank last 2 bit from second cell

        l = (mask & 0b00001111) << 4; // get last four bit, shift to left
        r = (mask & 0b00110000) >> 4; // get first 2 bit, shift to right

        *(dataBytes + cellIndex + 1) |= l; // setting the buffers
        *(dataBytes + cellIndex + 2) |= r;

        break;
    case 3:
        *(dataBytes + cellIndex + 2) &= 0b00000011; //blank first 6 bits
        *(dataBytes + cellIndex + 2) |= (mask << 2);
        break;
    }
}

template <typename T>
uint8_t oneMask();
template <typename T>
uint8_t setSignMaskPositive();
template <typename T>
uint8_t dataMaxPositiveNormalMask();
template <typename T>
uint8_t dataMaxNegativeNormalMask();
template <typename T>
uint8_t dataMaxPositiveSubNormalMask();
template <typename T>
uint8_t dataMaxNegativeSubNormalMask();
template <typename T>
uint8_t dataSubNormalOneMask();
template <typename T>
float dataMaxNormalNumber();
template <typename T>
float dataMinSubNormalNumber();
template <typename T>
uint8_t positiveZeroMask();
template <typename T>
uint8_t negativeZeroMask();
template <typename T>
uint8_t scaleSubNormalOne();
template <typename T>
uint8_t scaleOne();

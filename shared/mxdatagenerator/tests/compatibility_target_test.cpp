// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

int main()
{
    // Linking this executable against roc::mxDataGenerator verifies the only
    // remaining contract of the temporary compatibility project: legacy build
    // integrations can still resolve and consume its empty interface target.
    return 0;
}

#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Demo script to generate command lines for profiles."""

from rrtest import get_test_commands


def show_profile_commands(profile_name):
    """Show all commands for a profile."""
    commands = get_test_commands(profile_name)

    print(f"\nProfile: {profile_name}")
    print("=" * 70)

    for framework, cmd_list in commands.items():
        print(f"\n{framework}:")
        for cmd in cmd_list:
            print(f"  {' '.join(cmd)}")


if __name__ == "__main__":
    for profile in ["smoke", "precheckin", "codecov"]:
        show_profile_commands(profile)

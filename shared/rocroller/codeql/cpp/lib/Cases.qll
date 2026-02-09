// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

import cpp

string upperCase() {
  // ROCK_ROLLING
  result = "^[A-Z]+(_[A-Z0-9]+)*$"
}

string camelCase() {
  // rockRolling
  result = "^[a-z][a-zA-Z0-9]*$"
}

string pascalCase() {
  // RockRolling
  result = "^(?:[A-Z][a-z0-9]*)+$"
}

string snakeCaseAtLeastOneUnderscore() {
  // rock_rolling
  result = "^[a-z]+(_[a-z0-9]+)+$"
}

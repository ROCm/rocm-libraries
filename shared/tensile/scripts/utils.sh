################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

ERR='\033[0;31m'
INFO='\033[0;32m'
NC='\033[0m' # No color
echoerr() { echo -e "${ERR}$@${NC}" 1>&2; }
echoinfo() { echo -e "${INFO}$@${NC}" 1>&2; }

convert_comma_separated_to_array() {
    local var="$1"
    if [[ "$var" == *,* ]]; then
        IFS=',' read -r -a array <<< "$var"
        echo "${array[@]}"
    else
        echo "$var"
    fi
}

assert_envvar_exists() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echoerr "Error: Environment variable $var_name is not set, see \`--help\`."
        exit 1
    fi
}

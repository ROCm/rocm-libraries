# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

HIP_FILE=$1

if [[ "$HIP_FILE" =~ .*/src/device/.*\.h ]]; then
  perl -pi -e 's/(template<typename T, typename RedOp(?:, typename Proto)?)(, bool isNetOffload.*?)?>/\1, int COLL_UNROLL\2>/g' "$HIP_FILE"
  perl -pi -e 's/(ProtoSimple<[^,]*?,[^,]+?)>/\1, COLL_UNROLL>/g' "$HIP_FILE"
  perl -pi -e 's/(runRing<T.*?)((, (true|false))?>\()/\1, COLL_UNROLL\2/g' "$HIP_FILE"
  perl -pi -e 's/(runTreeUpDown<T.*?)>\(/\1, COLL_UNROLL>(/' "$HIP_FILE"
  perl -pi -e 's/(runTreeSplit<T.*?)>\(/\1, COLL_UNROLL>(/' "$HIP_FILE"
  sed -i "s/\\(struct RunWorkColl<ncclFunc[^>]*\\)>*/\\1, COLL_UNROLL>/" "$HIP_FILE"
  sed -i "s/\\(struct RunWorkBatch<ncclFunc[^>]*\\)>*/\\1, COLL_UNROLL>/" "$HIP_FILE"

  echo "Added COLL_UNROLL template argument to $HIP_FILE"
fi
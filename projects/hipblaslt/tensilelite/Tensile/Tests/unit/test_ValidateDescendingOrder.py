################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################


from Tensile.Components.CustomSchedule import ScheduleInfo
from Tensile.Components.CMSValidator import verify_ascending_order

class TestValidateDescendingOrder:
    def test_non_descending_order(self):
        """
        Test of the rule that instructions in each category
        appear in non-descending order
        """

        sched = ScheduleInfo(
            None, None, {"P": [[3, 2, 1]]}, None, None, None, None
        )
        status, message = verify_ascending_order(sched)

        expected = "Non-descending-order rule failed, schedule key 'P', sequence [3, 2, 1]: value 2 at index 1 is less than 3 at index 0."
        assert status == False
        assert message == expected

        sched = ScheduleInfo(
            None, None, {"P": [[1, 1, 2]]}, None, None, None, None
        )
        status, message = verify_ascending_order(sched)
        assert status == True


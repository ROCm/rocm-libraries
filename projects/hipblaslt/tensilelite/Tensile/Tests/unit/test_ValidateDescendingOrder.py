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


from Tensile.Components.CMSValidator import verify_ascending_order
from cms_validation_base import CMSValidationTestBase

class TestValidateDescendingOrder(CMSValidationTestBase):
    needs_timeline = False

    def validation_function(self, sched, kernel_dict, codePathIdx, timeline=None):
        return verify_ascending_order(sched, kernel_dict, codePathIdx)

    def test_non_descending_order_failure(self):
        """
        Test that validation fails when instructions appear in descending order.
        """
        optSchedule = {"P": [[3, 2, 1]]}
        expected = "Non-descending-order rule failed, schedule key 'P', sequence [3, 2, 1]: value 2 at index 1 is less than 3 at index 0."
        self.validate(optSchedule, [], 1, None, None, 0, expected_message=expected)

    def test_non_descending_order_success(self):
        """
        Test that validation passes when instructions appear in non-descending order.
        """
        optSchedule = {"P": [[1, 1, 2]]}
        self.validate(optSchedule, [], 1, None, None, 0)


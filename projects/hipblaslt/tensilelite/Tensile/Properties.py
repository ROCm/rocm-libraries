################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import hash_objs, state


class Property:

    @classmethod
    def FromOriginalState(cls, d):
        return cls(d.get('type'), d.get('index'), d.get('value'))

    def __init__(self, tag=None, index=None, value=None):
        self._tag = tag
        self._index = index
        self._value = value

    @property
    def tag(self):
        return self._tag

    @property
    def index(self):
        return self._index

    @property
    def value(self):
        return self._value

    def state(self):
        rv = {'type': self.tag}
        if self.index is not None: rv['index'] = state(self.index)
        if self.value is not None: rv['value'] = state(self.value)
        return rv

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.tag   == other.tag   and \
               self.value == other.value and \
               self.index == other.index

    def __hash__(self):
        #return hash(self.tag) ^ hash(self.value) ^ hash(self.index)
        return hash_objs(self.tag, self.value, self.index)


class Predicate(Property):

    @classmethod
    def And(cls, predicates):
        predicates = tuple(predicates)
        if len(predicates) == 0:
            return cls('TruePred')
        if len(predicates) == 1:
            return predicates[0]
        return cls('And', value=predicates)

    @classmethod
    def Or(cls, predicates):
        predicates = tuple(predicates)
        if len(predicates) == 0:
            return cls('TruePred')
        if len(predicates) == 1:
            return predicates[0]
        return cls('Or', value=predicates)

    def __lt__(self, other):
        # Ensure TruePred appears last.
        if other.tag == 'TruePred':
            if self.tag == 'TruePred':
                return False
            return True
        if self.tag == 'TruePred':
            return False

        selfValue = self.value
        otherValue = other.value

        if type(selfValue) is dict:
            selfTup = {v for _, v in selfValue.items()}
            selfValue = selfTup

        if type(otherValue) is dict:
            otherTup = {v for _, v in otherValue.items()}
            otherValue = otherTup

        # If neither is a TruePred then just use the default comparison.
        return (self.tag, self.index, selfValue) < (other.tag, other.index, otherValue)

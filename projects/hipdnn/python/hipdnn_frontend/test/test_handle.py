# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU tests for the Handle lifecycle and stream API."""

import pytest

import hipdnn_frontend as hipdnn


@pytest.mark.gpu
class TestHandle:
    """Tests for handle creation, stream access, and destruction."""

    @pytest.mark.parametrize("factory", [hipdnn.Handle, hipdnn.create_handle])
    def test_construct_returns_valid_pointer(self, factory):
        """Both the Handle ctor and create_handle() yield a valid pointer."""
        handle = factory()
        assert int(handle) != 0

    @pytest.mark.parametrize("factory", [hipdnn.Handle, hipdnn.create_handle])
    def test_construct_with_stream_binds_stream(self, factory):
        """Both entry points bind the handle to the given stream."""
        handle = factory(0)
        assert handle.get_stream() == 0

    @pytest.mark.parametrize(
        "set_stream, get_stream",
        [
            (lambda h, s: h.set_stream(s), lambda h: h.get_stream()),
            (hipdnn.set_stream, hipdnn.get_stream),
        ],
        ids=["method", "module_fn"],
    )
    def test_set_and_get_stream(self, set_stream, get_stream):
        """set_stream()/get_stream() round-trip via both method and module fn."""
        handle = hipdnn.create_handle()
        set_stream(handle, 0)
        assert get_stream(handle) == 0

    def test_destroy_handle(self):
        """destroy_handle() invalidates the handle (repr shows destroyed)."""
        handle = hipdnn.create_handle()
        hipdnn.destroy_handle(handle)
        assert "destroyed" in repr(handle)

    def test_get_stream_after_destroy_raises(self):
        """Accessing the stream after destroy raises RuntimeError."""
        handle = hipdnn.create_handle()
        hipdnn.destroy_handle(handle)
        with pytest.raises(RuntimeError):
            handle.get_stream()

    def test_set_stream_after_destroy_raises(self):
        """Setting the stream after destroy raises RuntimeError."""
        handle = hipdnn.create_handle()
        hipdnn.destroy_handle(handle)
        with pytest.raises(RuntimeError):
            handle.set_stream(0)

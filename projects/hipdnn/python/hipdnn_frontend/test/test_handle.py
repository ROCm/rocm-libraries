# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU tests for the Handle lifecycle and stream API."""

import pytest

import hipdnn_frontend as hipdnn


@pytest.mark.gpu
class TestHandle:
    """Tests for handle creation, stream access, and destruction."""

    def test_create_handle_default(self):
        """create_handle() returns a usable handle with a valid pointer."""
        handle = hipdnn.create_handle()
        assert int(handle) != 0

    def test_handle_default_constructor(self):
        """Handle() constructs a handle directly."""
        handle = hipdnn.Handle()
        assert int(handle) != 0

    def test_handle_stream_constructor(self):
        """Handle(stream) constructs a handle bound to the given stream."""
        handle = hipdnn.Handle(0)
        assert handle.get_stream() == 0

    def test_create_handle_with_stream(self):
        """create_handle(stream) binds the handle to the given stream."""
        handle = hipdnn.create_handle(0)
        assert handle.get_stream() == 0

    def test_set_and_get_stream_method(self):
        """set_stream()/get_stream() round-trip on the handle object."""
        handle = hipdnn.create_handle()
        handle.set_stream(0)
        assert handle.get_stream() == 0

    def test_set_and_get_stream_module_fns(self):
        """Module-level set_stream()/get_stream() operate on a handle."""
        handle = hipdnn.create_handle()
        hipdnn.set_stream(handle, 0)
        assert hipdnn.get_stream(handle) == 0

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

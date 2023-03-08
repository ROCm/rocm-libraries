import matrix_utils
import msgpack


def test_input_basic():
    x = 128
    y = 16
    test_val = {"sizes": [x, y], "data": [0] * (x * y)}

    with open("data.msgpack", "wb") as outfile:
        packed = msgpack.packb(test_val)
        outfile.write(packed)

    test = matrix_utils.loadMatrix("data.msgpack")

    assert test.shape == [16, 128]  # Should be loaded transposed.

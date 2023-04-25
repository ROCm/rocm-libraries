import matrix_utils
import msgpack


def test_input_basic():
    x = 16
    y = 128
    test_val = {"sizes": [x, y], "data": list(range(x * y))}

    with open("data.msgpack", "wb") as outfile:
        packed = msgpack.packb(test_val)
        outfile.write(packed)

    test = matrix_utils.loadMatrix("data.msgpack")

    assert test.shape == (16, 128)
    assert test[0, 0] == 0
    assert test[0, 1] == 16
    assert test[1, 0] == 1

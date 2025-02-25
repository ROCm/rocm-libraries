import matrix_utils
import msgpack


def test_input_basic(tmp_path_factory):
    x = 16
    y = 128
    test_val = {"sizes": [x, y], "data": list(range(x * y))}

    out_dir = tmp_path_factory.mktemp("test_matrixIO_input")
    f = out_dir / "data.msgpack"
    f.write_bytes(msgpack.packb(test_val))

    test = matrix_utils.loadMatrix(f)

    assert test.shape == (16, 128)
    assert test[0, 0] == 0
    assert test[0, 1] == 16
    assert test[1, 0] == 1

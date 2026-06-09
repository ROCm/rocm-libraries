# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for dnn-benchmark TOML config files."""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.cli.config_file import apply_config_file
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.reporting.reporter import Reporter


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _parse_with_config(argv: list[str]):
    args = create_parser(suppress_defaults=True).parse_args(argv)
    apply_config_file(args)
    return args


def test_cli_abbreviations_are_treated_as_explicit_overrides(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["from_config.json"]
warmup = 3
iters = 7
""",
    )

    args = _parse_with_config(
        ["--config", str(config), "--graph", "from_cli.json", "--iter", "11"]
    )

    assert args.graph == ["from_cli.json"]
    assert args.warmup == 3
    assert args.iters == 11


def test_config_populates_args_when_cli_does_not_override(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["graphs/a.json", "graphs/b.json"]
warmup = 3
iters = 7
seed = 42
validate = "none"
metrics_tier = "off"

[[engines]]
id = 1
plugin_path = "/plugins/b"

[[engines]]
id = 1
plugin_path = "/plugins/a"
""",
    )

    args = _parse_with_config(["--config", str(config)])

    assert args.graph == [
        str(tmp_path / "graphs/a.json"),
        str(tmp_path / "graphs/b.json"),
    ]
    assert args.warmup == 3
    assert args.iters == 7
    assert args.seed == 42
    assert args.metrics_tier == "off"
    assert args.engine == [1, 1]
    assert args.plugin_path == [Path("/plugins/b"), Path("/plugins/a")]


def test_cli_scalars_override_config_values(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["from_config.json"]
iters = 7
""",
    )

    args = _parse_with_config(
        ["--config", str(config), "--graph", "from_cli.json", "--iters", "11"]
    )

    assert args.graph == ["from_cli.json"]
    assert args.iters == 11


def test_cli_engine_replaces_config_engine_matrix(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]

[[engines]]
id = 2
plugin_path = "/plugins/b"

[[engines]]
id = 1
plugin_path = "/plugins/a"
""",
    )

    args = _parse_with_config(["--config", str(config), "--engine", "9,8"])

    assert args.engine == [9, 8]
    assert args.plugin_path is None
    assert not hasattr(args, "_config_engine_names")


def test_comparison_table_is_rejected_as_unknown_field(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]

[comparison]
baseline = "missing"
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match="Unknown config field: comparison"):
        apply_config_file(args)


def test_engine_config_uses_ids_without_display_labels(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]

[[engines]]
id = 2

[[engines]]
id = 1
""",
    )

    args = _parse_with_config(["--config", str(config)])

    assert args.engine == [2, 1]
    assert not hasattr(args, "_config_engine_names")


def test_every_engine_sets_plugin_path_when_any_engine_does(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]

[[engines]]
id = 1
plugin_path = "/plugins/a"

[[engines]]
id = 2
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match="Every config engine must set plugin_path"):
        apply_config_file(args)


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ("itres = 1000", "itres"),
        ("[profiling]\nenabled = true", "profiling"),
    ],
)
def test_unknown_top_level_config_fields_are_rejected(
    tmp_path: Path, body: str, field: str
) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        f"""
version = 1
graphs = ["g.json"]
{body}
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match=f"Unknown config field: {field}"):
        apply_config_file(args)


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ('plugin_pat = "/plugins"', "plugin_pat"),
        ('label = "baseline"', "label"),
        ('name = "baseline"', "name"),
    ],
)
def test_unknown_engine_config_fields_are_rejected(
    tmp_path: Path, body: str, field: str
) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        f"""
version = 1
graphs = ["g.json"]

[[engines]]
id = 1
{body}
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match=f"Unknown config engine 0 field: {field}"):
        apply_config_file(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", '"pytoch"'),
        ("validate", '"torch"'),
        ("metrics_tier", '"full"'),
        ("emit_trace", '"json"'),
        ("pmc", '"everything"'),
    ],
)
def test_invalid_config_choice_values_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        f"""
version = 1
graphs = ["g.json"]
{field} = {value}
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match=f"Config field '{field}' must be one of"):
        apply_config_file(args)


def test_bool_is_rejected_for_integer_config_fields(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]
iters = true
""",
    )
    args = create_parser(suppress_defaults=True).parse_args(["--config", str(config)])

    with pytest.raises(ValueError, match="Config field 'iters' must be int"):
        apply_config_file(args)


def test_config_paths_are_relative_to_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = _write_config(
        config_dir / "bench.toml",
        """
version = 1
graphs = ["../graphs/g.json"]
output = "results/out.json"
plugin_path = "../plugins"
profiling_output_dir = "profiles"
""",
    )

    args = _parse_with_config(["--config", str(config)])

    assert args.graph == [str(config_dir / "../graphs/g.json")]
    assert args.output == config_dir / "results/out.json"
    assert args.plugin_path == [config_dir / "../plugins"]
    assert args.profiling_output_dir == config_dir / "profiles"


def test_sample_configs_parse_and_reference_existing_graphs() -> None:
    root = Path(__file__).resolve().parents[3]

    full_config = root / "sample_configs" / "config.toml.example"
    full_args = _parse_with_config(["--config", str(full_config)])

    assert full_args.graph
    for graph in full_args.graph:
        assert Path(graph).exists()

    basic_config = root / "sample_configs" / "basic.toml.example"
    graph = root / "graphs" / "sample_conv_fwd.json"
    basic_args = _parse_with_config(
        ["--config", str(basic_config), "--graph", str(graph)]
    )

    assert basic_args.graph == [str(graph)]
    assert basic_args.warmup == 10
    assert basic_args.iters == 100
    assert basic_args.verbose is True
    assert basic_args.plugin_path == [Path("/opt/rocm/lib/hipdnn_plugins/engines")]


def test_invalid_config_backend_errors_before_graph_resolution(
    tmp_path: Path, capsys
) -> None:
    config = _write_config(
        tmp_path / "bench.toml",
        """
version = 1
graphs = ["g.json"]
backend = "pytoch"
""",
    )
    main_module = importlib.import_module("dnn_benchmarking.cli.main")

    with (
        patch.object(main_module, "_resolve_graphs") as mock_resolve,
        patch("sys.argv", ["dnn-benchmark", "--config", str(config)]),
        pytest.raises(SystemExit) as exc,
    ):
        main_module.main()

    assert exc.value.code == 2
    mock_resolve.assert_not_called()
    assert "Config field 'backend' must be one of" in capsys.readouterr().err


@patch("dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark")
def test_main_uses_config_graphs_and_builds_suite_config(
    mock_benchmark: MagicMock, tmp_path: Path
) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
    config = _write_config(
        tmp_path / "bench.toml",
        f"""
version = 1
graphs = ["{graph}"]
warmup = 1
iters = 2

[[engines]]
id = 2

[[engines]]
id = 1
""",
    )
    mock_benchmark.return_value = 0

    main_module = importlib.import_module("dnn_benchmarking.cli.main")

    with patch("sys.argv", ["dnn-benchmark", "--config", str(config)]):
        rc = main_module.main()

    assert rc == 0
    suite_config = mock_benchmark.call_args.kwargs["config"]
    assert suite_config.warmup_iters == 1
    assert suite_config.benchmark_iters == 2
    assert suite_config.engine_filter == [2, 1]


def test_missing_graph_without_config_errors(capsys) -> None:
    main_module = importlib.import_module("dnn_benchmarking.cli.main")

    with (
        patch("sys.argv", ["dnn-benchmark"]),
        pytest.raises(SystemExit) as exc,
    ):
        main_module.main()

    assert exc.value.code == 2
    assert "--graph is required" in capsys.readouterr().err

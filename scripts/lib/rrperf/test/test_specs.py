import rrperf


def test_spec_basic():
    test = rrperf.specs.MachineSpecs(hostname="test_host")
    assert test.hostname == "test_host"


def test_spec_yaml(tmp_path):
    test = rrperf.specs.MachineSpecs(hostname="test_host")
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text(str(test))
    loaded = rrperf.specs.load_machine_specs(yaml_file)
    assert loaded.hostname == "test_host"

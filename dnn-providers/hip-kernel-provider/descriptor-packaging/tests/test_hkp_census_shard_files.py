"""Files a census shard reads by bare name sit beside its build-tree CTest file.

Runs the real hkp_reserve_census_shard() and hkp_finalize_census_install() from
HkpPackaging.cmake under ``cmake -P``. A shard entry passes ``--test-config <name>.toml``
by bare name, so the shard runs from the build tree only if the reserved TOML is copied
beside it; a shard that names a TOML nothing places must fail configuration.
"""

import subprocess
from pathlib import Path

import pytest

_CMAKE_DIR = Path(__file__).resolve().parents[1] / "cmake"
_PROVIDER_CMAKE_DIR = _CMAKE_DIR.parents[1] / "cmake"

pytestmark = pytest.mark.quick

_TOML = "hipkernel_Example.toml"


def _finalize(cmake, tmp_path, entry, reserve_args):
    """Reserve the gfx950 shard, stage one external entry, finalize; return the run."""
    build = tmp_path / "build"
    common = tmp_path / "CTestTestfile.cmake"
    common.write_text("")
    script = tmp_path / "finalize.cmake"
    script.write_text(
        f'list(APPEND CMAKE_MODULE_PATH "{_CMAKE_DIR.as_posix()}" '
        f'"{_PROVIDER_CMAKE_DIR.as_posix()}")\n'
        "include(HkpPackaging)\n"
        'set(HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR "lib/hipdnn_plugins/engines")\n'
        'set(HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR "test_arch_content/hip-kernel-provider")\n'
        'set(HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR "arch_content/hip-kernel-provider")\n'
        f'set(HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR "{build.as_posix()}")\n'
        f"hkp_reserve_census_shard(_dest gfx950 {reserve_args})\n"
        "set_property(GLOBAL APPEND_STRING PROPERTY "
        f"EXTERNAL_TEST_INSTALL_STAGING_hkp_census_gfx950 [==[{entry}\n]==])\n"
        f'hkp_finalize_census_install(COMMON_TEST_FILE "{common.as_posix()}" '
        'BINDIR "bin" PLUGIN_ENGINE_DIR "lib/hipdnn_plugins/engines")\n'
    )
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    return result, build / "census" / "gfx950"


def _entry(name):
    return f'add_test(suite "../../bin/run" "--test-config" {name})'


def test_reserved_file_lands_beside_the_shard(cmake, tmp_path):
    source = tmp_path / "config" / _TOML
    source.parent.mkdir()
    source.write_text("[meta]\nversion = 1\n")

    result, shard = _finalize(
        cmake, tmp_path, _entry(f'"{_TOML}"'), f'FILES "{source.as_posix()}"'
    )

    assert result.returncode == 0, result.stderr
    assert (shard / "CTestTestfile.cmake").is_file()
    assert (shard / _TOML).read_bytes() == source.read_bytes()


# The two quoting forms the external-integration emitters write.
@pytest.mark.parametrize("quoted", [f'"{_TOML}"', f"[[{_TOML}]]"])
def test_named_toml_nothing_places_fails_configuration(cmake, tmp_path, quoted):
    result, _ = _finalize(cmake, tmp_path, _entry(quoted), "")

    assert result.returncode != 0
    assert f"the gfx950 census shard names {_TOML}" in result.stderr


def test_unrecognised_reservation_argument_fails(cmake, tmp_path):
    result, _ = _finalize(cmake, tmp_path, _entry(f'"{_TOML}"'), "FILE x.txt")

    assert result.returncode != 0
    assert "hkp_reserve_census_shard: unrecognised argument(s): FILE;x.txt" in (
        result.stderr
    )

from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    @classmethod
    def has_ext_modules(foo):
        return True


def wheel_name(**kwargs):
    # create a fake distribution from arguments
    dist = BinaryDistribution(attrs=kwargs)
    # finalize bdist_wheel command
    bdist_wheel_cmd = dist.get_command_obj('bdist_wheel')
    bdist_wheel_cmd.ensure_finalized()
    # assemble wheel file name
    distname = bdist_wheel_cmd.wheel_dist_name
    tag = '-'.join(bdist_wheel_cmd.get_tag())
    return f'{distname}-{tag}.whl'

# Version updates - IMP: Change in version requires to match top level CMakeLists.txt
## * getrocALWheelname.py
## * setup.py
print(wheel_name(name='amd-rocal', version='2.3.0'), end='')

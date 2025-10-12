# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import yaml
from pathlib import Path

try:
    DEFAULT_YAML_LOADER = yaml.CSafeLoader
except:
    print('CSafeLoader is not installed.')
    DEFAULT_YAML_LOADER = yaml.SafeLoader

def parse_general(loader: yaml.Loader):
    if loader.check_event(yaml.MappingStartEvent):
        return parse_mapping(loader)
    elif loader.check_event(yaml.SequenceStartEvent):
        return parse_sequence(loader)
    elif loader.check_event(yaml.ScalarEvent):
        return parse_scalar(loader)

def parse_sequence(loader: yaml.Loader):
    ret = []
    #pop sequence start event
    loader.get_event()
    while not loader.check_event(yaml.SequenceEndEvent):
        ret.append(parse_general(loader))
    #pop sequence end event
    loader.get_event()
    return ret

def parse_mapping(loader: yaml.Loader):
    ret = {}
    k, v = None, None
    #pop mapping start event
    loader.get_event()
    while not loader.check_event(yaml.MappingEndEvent):
        if k is None:
            k = parse_scalar(loader)
        elif v is None:
            v = parse_general(loader)
            ret[k] = v
            k, v = None, None

    #pop mapping end event
    loader.get_event()
    return ret

def is_float(value):
    if not value:
        return False
    first_char = value[0]
    if not (first_char in '+-.' or first_char.isdigit()):
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_scalar(loader: yaml.Loader):
    assert loader.check_event(yaml.ScalarEvent)
    evt = loader.get_event()
    value: str = evt.value

    if not value:
        if not evt.style:
            return None
        return value

    first_char = value[0]
    if first_char in '+-' or first_char.isdigit():
        stripped = value.lstrip('+-')
        if stripped.isdigit():
            return int(value)
        elif is_float(value):
            return float(value)

    value_folded = value.casefold()

    if value_folded in ('true', 'yes'):
        return True
    elif value_folded in ('false', 'no'):
        return False
    elif value_folded in ('null', '~'):
        if not evt.style:
            return None

    return value

def load_yaml_stream(yaml_path: Path, loader_type: yaml.Loader):
    with open(yaml_path, 'r') as f:
        loader = loader_type(f)
        assert loader.check_event(yaml.StreamStartEvent)
        loader.get_event()
        assert loader.check_event(yaml.DocumentStartEvent)
        loader.get_event()
        logic = parse_general(loader)
        assert loader.check_event(yaml.DocumentEndEvent)
        loader.get_event()
        assert loader.check_event(yaml.StreamEndEvent)
        return logic

def load_yaml_sequence_item(yaml_path: Path, loader_type: yaml.Loader, idx: int):
    with open(yaml_path, 'r') as f:
        loader = loader_type(f)
        assert loader.check_event(yaml.StreamStartEvent)
        loader.get_event()
        assert loader.check_event(yaml.DocumentStartEvent)
        loader.get_event()

        # assume the root element is a sequence
        if not loader.check_event(yaml.SequenceStartEvent):
            raise RuntimeError('Root of YAML is not a sequence')

        loader.get_event()
        cur_idx = 0
        ret = None

        while not loader.check_event(yaml.SequenceEndEvent):
            obj = parse_general(loader)

            if cur_idx == idx:
                ret = obj
                break

            cur_idx += 1

        return ret

def load_yaml_dict_item(yaml_path: Path, loader_type: yaml.Loader, key: str):
    with open(yaml_path, 'r') as f:
        loader = loader_type(f)
        assert loader.check_event(yaml.StreamStartEvent)
        loader.get_event()
        assert loader.check_event(yaml.DocumentStartEvent)
        loader.get_event()

        # assume the root element is a map
        if not loader.check_event(yaml.MappingStartEvent):
            raise RuntimeError('Root of YAML is not a map')

        loader.get_event()
        k, v = None, None

        while not loader.check_event(yaml.MappingEndEvent):
            if k is None:
                k = parse_scalar(loader)
            else:
                value = parse_general(loader)

                if k == key:
                    v = value
                    break
                k = None

        return v

def load_logic_gfx_arch(yaml_path: Path, loader_type: yaml.Loader = DEFAULT_YAML_LOADER):
    try:
        GFX_ARCH_IDX = 2
        arch = load_yaml_sequence_item(yaml_path, loader_type, GFX_ARCH_IDX)

        if isinstance(arch, dict):
            return arch['Architecture']
        else:
            return arch
    except RuntimeError as e:
        return load_yaml_dict_item(yaml_path, loader_type, 'ArchitectureName')

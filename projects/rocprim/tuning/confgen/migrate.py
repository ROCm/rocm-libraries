#!/usr/bin/env python3

import logging
import argparse
import os
import re

import jinja2

log = logging.getLogger("confgen.migrate")


# Marker wrapped around a config parameter name when the config-constructor
# template is rendered. Chosen to never appear in real C++ source.
_SENTINEL = "\x00"


def try_make_int(text: str):
    try:
        return int(text)
    except:
        return text.strip()


class RecordingConfig(dict):
    """A stand-in for the ``config`` object passed to the config-constructor
    template. Any parameter the template reads (``config.block_size_x``,
    ``config.ipt``, ``config['__ipt__']``, ...) resolves to a sentinel-wrapped
    copy of its own name, so the rendered constructor becomes a pattern we can
    turn into a regex and match real configs against.
    """

    def __missing__(self, key):
        return f"{_SENTINEL}{key}{_SENTINEL}"


def render_config_constructor(template_file: str) -> str:
    """Render only the ``config_constructor`` block of a config header template,
    with each tuned parameter replaced by a sentinel-wrapped name.
    """
    template_dir = os.path.dirname(os.path.abspath(template_file))
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )
    template = env.get_template(os.path.basename(template_file))

    if "config_constructor" not in template.blocks:
        raise ValueError(
            f"Template '{template_file}' does not define a 'config_constructor' block."
        )

    context = template.new_context({"config": RecordingConfig()})
    return "".join(template.blocks["config_constructor"](context))


def build_config_regex(rendered: str) -> re.Pattern:
    """Turn a rendered (sentinel-marked) config constructor into a regex whose
    named groups capture the tuned parameter values.

    Whitespace is stripped so that fixed sub-configs like
    '(1 << 17) + 70000' match regardless of formatting. Parameters that appear
    more than once become back-references so all occurrences must be equal.
    """
    compact = re.sub(r"\s+", "", rendered)
    parts = re.split(f"{_SENTINEL}([^{_SENTINEL}]+){_SENTINEL}", compact)

    pattern = ""
    seen: set[str] = set()
    for index, part in enumerate(parts):
        if index % 2 == 0:
            # Literal chunk of the constructor.
            pattern += re.escape(part)
        else:
            # A tuned parameter name.
            name = part
            if name in seen:
                pattern += f"(?P={name})"
            else:
                seen.add(name)
                pattern += rf"(?P<{name}>[^,{{}}]+)"
    return re.compile(pattern)


def main():
    cli = argparse.ArgumentParser(
        description="Annotate a legacy config header with '// CONFIG:' and "
        "'// TARGET:' comments so it can be ingested by generate.py."
    )
    cli.add_argument("--config", "-c", required=True, help="Config header to migrate.")
    cli.add_argument(
        "--template-file",
        "-t",
        required=True,
        help="Path to the matching 'confgen/templates/device_<alg>.h.jinja2'. Its "
        "'config_constructor' block is used to map constructor values back to "
        "Kernel Tuner parameter names.",
    )

    args = cli.parse_args()

    config_regex = build_config_regex(render_config_constructor(args.template_file))
    log.debug(f"Config pattern: {config_regex.pattern}")

    modified_lines: list[str] = []
    with open(args.config, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            emit_current_line = True

            # Detect target guard.
            # Assume this is a top level template, so we don't have to do any other fancy matching.
            if re.match(r"^template", line):
                for search_index in range(i, len(lines)):
                    comp_target = re.match(
                        r"comp_target<gen::(?P<gen>\w+), target_arch::(?P<arch>\w+), gpu::(?P<gpu>\w+), rep::(?P<rep>\w+)>>",
                        lines[search_index].strip(),
                    )
                    if comp_target:
                        target = {
                            "gen": comp_target["gen"],
                            "arch": comp_target["arch"],
                            "gpu": comp_target["gpu"],
                            "rep": comp_target["rep"],
                        }
                        log.debug(f"Found target: {target}")
                        modified_lines.append(f"// TARGET: {target}\n")
                        break

            # Detect type guard (any variable names).
            types_match = re.match(r"^ +// Based on (?P<types>.+)$", line)

            if types_match is not None:
                # Parse "a = x, b = y, c = z" into {"a": "x", "b": "y", "c": "z"}
                config = {}
                for pair in types_match["types"].split(", "):
                    name, _, value = pair.partition(" = ")
                    config[name.strip()] = value.strip()

                return_found = False
                return_complete = False
                return_string = ""
                for search_index in range(i, len(lines)):
                    prefix_regex = r"^.*?return"
                    postfix_regex = r"^([^;]*);.*$"
                    search_line = lines[search_index]

                    if (not return_found) and re.match(prefix_regex, search_line):
                        search_line = re.sub(prefix_regex, "", search_line)
                        return_found = True

                    if return_found and re.match(postfix_regex, search_line):
                        search_line = re.sub(postfix_regex, r"\1", search_line)
                        return_complete = True

                    if return_found:
                        return_string += search_line.strip()
                    if return_complete:
                        break

                assert return_complete

                # Map the constructor values back to parameter names by matching
                # the concrete config against the template-derived pattern.
                compact_return = re.sub(r"\s+", "", return_string)
                match = config_regex.match(compact_return)
                if match is None:
                    log.warning(
                        f"Could not match config against template, leaving unchanged: "
                        f"{return_string}"
                    )
                else:
                    config |= {
                        name: try_make_int(value)
                        for name, value in match.groupdict().items()
                    }
                    log.debug(f"Found config: {config}")

                    # Replace current line
                    emit_current_line = False
                    modified_lines.append(f"    // CONFIG: {config}\n")

            # Emit line if we need to
            if emit_current_line:
                modified_lines.append(line)

    with open(args.config, "w") as file:
        file.write("".join(modified_lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
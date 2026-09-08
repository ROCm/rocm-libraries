#!/usr/bin/env python3

import logging
import argparse
import re

log = logging.getLogger("confgen.migrate")


def try_make_int(text: str):
    try:
        return int(text)
    except:
        return text.strip()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", "-c")
    cli.add_argument("--variable", "-v", action="append")

    args = cli.parse_args()

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

            # Detect type guard.
            # Assume line has 'Based on' string
            types = re.match(
                r"^ +// Based on key_type = (?P<key_type>[\w:]+), value_type = (?P<value_type>[\w:]+)",
                line,
            )
            if types is not None:
                # Inject types into config
                config = {}
                key_type = types["key_type"]
                if key_type:
                    config["key_type"] = key_type
                value_type = types["value_type"]
                if value_type:
                    config["value_type"] = value_type

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

                # Assume the arguments to the constructor are integers only.
                # We can easily get the flattened arguments by splitting by comma:
                #   { a, b, {c, {d}, e}, f } => a, b, c, d, e, f
                # We can then zip it with the requested value names to re-construct
                # the config-annotation.
                assert return_complete
                return_string = re.sub(r"^\w*", "", return_string)
                return_string = re.sub(r"[\{\}a-zA-Z_ ]", "", return_string)
                return_values = [try_make_int(x) for x in return_string.split(",")]
                config |= dict(zip(args.variable, return_values))
                log.debug(f"Found config: {config}")

                # Replace current line
                emit_current_line = False
                modified_lines.append(f"    // CONFIG: {config}\n")

            # Emit line if we need to
            if emit_current_line:
                modified_lines.append(line)
        pass

    with open(args.config, "w") as file:
        file.write("".join(modified_lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

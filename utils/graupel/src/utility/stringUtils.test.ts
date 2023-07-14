import StringUtils from "./stringUtils";
import { describe, expect, test } from "@jest/globals";

describe("stringUtils", () => {
  let cases = [
    ["path/to/file.s", "s"],
    ["path/to/file.multi.yaml", "yaml"],
  ];
  test.each(cases)(StringUtils.getFileExtension.name, (arg, expected) => {
    expect(StringUtils.getFileExtension(arg)).toEqual(expected);
  });

  cases = [["camelCaseString", "Camel Case String"]];
  test.each(cases)(StringUtils.beautifyCamelCase.name, (arg, expected) => {
    expect(StringUtils.beautifyCamelCase(arg)).toEqual(expected);
  });
});

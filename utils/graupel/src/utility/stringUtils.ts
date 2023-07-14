import { CY_CLASSES } from "../constants";

module StringUtils {
  /**
   * Converts "camelCase" into "Camel Case"
   * @param camelCase
   * @returns beautified string
   */
  export function beautifyCamelCase(camelCase: string) {
    var readableCase = camelCase.replace(/([a-z])([A-Z])/g, "$1 $2");

    readableCase = readableCase.replace(/\b\w/g, function (match) {
      return match.toUpperCase();
    });

    return readableCase;
  }

  /**
   * Gets file extension from path
   * @param path 
   * @returns extension (no "." prefix)
   */
  export function getFileExtension(path: string) {
    if (!path) throw Error("Empty path provided");
    const parts = path.split(".");
    if (parts.length > 1) {
      return parts.pop();
    }
    throw Error(`No extension in path ${path}`);
  }

  /**
   * Parses common abbreviations for coordinates and control
   * @param input abbrivation + numerical, e.g. "coor1"
   * @param func function to call with selector
   */
  export function smartId(input: string, func: Function) {
    function subgraphDiscriminator(input: string) {
      const controlOptions = ["con", "cn", "ct", "cnt"];
      const coordinateOptions = ["coo", "cr", "cd", "crd"];

      const controlCount = controlOptions.filter((e) =>
        input.includes(e)
      ).length;
      const coordCount = coordinateOptions.filter((e) =>
        input.includes(e)
      ).length;

      return controlCount > coordCount
        ? CY_CLASSES.CONTROL
        : CY_CLASSES.COORDINATES;
    }

    const [_, prefix, suffix] = input.match(/^([a-zA-Z]+)(\d+)$/); // abc123 -> [abc123, abc, 123]
    const subgraphType = subgraphDiscriminator(prefix);

    console.log(`Parsed ${input} to ${subgraphType + suffix}`);
    return func(`[id="${subgraphType + suffix}"]`);
  }
}

export default StringUtils;

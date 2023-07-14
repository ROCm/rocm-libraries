import * as Deserializer from "./deserializer.js";
import yaml from "js-yaml";

const testOutputPath = "./src/model/helper/";

async function readFileAsText(filePath) {
  return await import("fs").then((fs) => {
    return fs.readFileSync(filePath, "utf8");
  });
}

async function writeJsonToFile(filePath, data) {
  if (typeof filePath === "undefined") {
    return;
  }
  await import("fs").then((fs) => {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log("`writeFileAsText`: wrote to " + filePath);
  });
}

/**
 * Converts a YAML file to a JS object
 * @param {string} inputPath relative path to input YAML file
 * @param {CallableFunction} conversionFunc conversion function, with rocRoller graph object as parameter and JS object as return
 */
export async function yamlTest(inputPath, conversionFunc) {
  try {
    const text = await readFileAsText(inputPath);
    const input = yaml.load(text);
    writeJsonToFile(testOutputPath + "yaml.json", input);
    const output = conversionFunc(input);
    writeJsonToFile(testOutputPath + "yamlOutput.json", output);
    return output;
  } catch (e) {
    console.error(e);
  }
}

/**
 * Converts a Assembly file to a JS object
 * @param {string} inputPath relative path to input Assembly file
 * @param {CallableFunction} conversionFunc conversion function, with rocRoller graph object as parameter and JS object as return
 */
export async function asmTest(inputPath, conversionFunc) {
  try {
    const text = await readFileAsText(inputPath);
    const startIdx = text.indexOf("---\n");
    const endIdx = text.indexOf("...\n");
    const input = yaml.load(text.substring(startIdx, endIdx));
    const graph = input["amdhsa.kernels"][0][".kernel_graph"];
    const output = conversionFunc(graph);
    writeJsonToFile(testOutputPath + "asmOutput.json", output);
    return output;
  } catch (e) {
    console.error(e);
  }
}

test("deserializer", async () => {
  const asmOutput = await asmTest(
    "./kernels/sample.s",
    Deserializer.convertRRtoCytoscape
  );
  const yamlOutput = await yamlTest(
    "./kernels/sample.yaml",
    Deserializer.convertRRtoCytoscape
  );
  // TODO: deep assert/equal asmOutput and yamlOutput
});

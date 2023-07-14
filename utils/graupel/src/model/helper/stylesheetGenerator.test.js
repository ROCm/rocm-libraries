import * as SSG from "./stylesheetGenerator.js";
import * as Deserializer from "./deserializer.js";
import * as StylesheetGenerator from "./stylesheetGenerator.js";
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

test("stylesheetGenerator", async () => {
  const keyedGroup = await asmTest(
    "./kernels/sample.s",
    Deserializer.convertRRtoCytoscape
  );
  const stylesheet = StylesheetGenerator.generateStylesheet(keyedGroup);
  writeJsonToFile(testOutputPath + "stylesheet.json", stylesheet);
});

test("genDistinctHues", () => {
  for (let numColours = 1; numColours <= 30; ++numColours) {
    let colours = [];
    const gen = SSG.genDistinctHues();
    for (let i = 0; i < numColours; ++i) {
      colours.push(gen.next().value);
    }
    colours = colours.sort((a, b) => a - b);
    colours.push(360);
    let minDiff = Infinity;
    for (let i = 0; i < colours.length - 1; i++) {
      minDiff = Math.min(colours[i + 1] - colours[i], minDiff);
    }
    // ensures distinctness of hues
    expect(minDiff).toBeGreaterThanOrEqual(360 / (numColours * 2));
  }
});
